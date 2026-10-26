"""Shared protocol and auditable artifacts for the real Dry Bean CPU benchmark."""
import os

THREADS = 4
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = str(THREADS)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'dry_bean_prepared.npz'
OUTPUT_DIR = ROOT / 'results_v2'
SPLIT_SEED = 42
SEEDS = (42, 123, 2026, 3407, 7777)
HIDDEN = (16, 32, 64)
INPUT_DIM, NUM_CLASSES = 16, 7
EPOCHS, BATCH_SIZE = 100, 256
LEARNING_RATE, CLIP_NORM = 0.0003, 1.0
BETA1, BETA2, EPSILON, WEIGHT_DECAY = 0.9, 0.999, 1e-8, 0.0
DIMS = (INPUT_DIM, *HIDDEN, NUM_CLASSES)
FRAMEWORKS = {'PyTorch': 'train_pytorch.py', 'TensorFlow': 'train_tensorflow.py',
              'PaddlePaddle': 'train_paddlepaddle.py', 'MindSpore': 'train_mindspore.py'}
EXPECTED_VERSIONS = {'PyTorch': '2.9.1', 'TensorFlow': '2.21.0',
                     'PaddlePaddle': '3.3.0', 'MindSpore': '2.10.0'}
METRIC_FIELDS = ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                 'test_cross_entropy', 'train_seconds', 'validation_seconds',
                 'training_plus_validation_seconds', 'test_seconds', 'total_seconds')
PROTOCOL = dict(protocol_version=3, architecture=list(DIMS), split_seed=SPLIT_SEED,
                epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                gradient_clip_global_norm=CLIP_NORM, optimizer='Adam', beta1=BETA1,
                beta2=BETA2, epsilon=EPSILON, weight_decay=WEIGHT_DECAY,
                threads_requested=THREADS, device='CPU', dtype='float32',
                initialization='NumPy PCG64 He-uniform, zero biases',
                selection='Final epoch, no test-based model selection')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError('Run python prepare_data.py first.')
    with np.load(DATA_PATH, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    if len(data['class_names']) != NUM_CLASSES or int(data['split_seed']) != SPLIT_SEED:
        raise ValueError('Unexpected dataset schema/split seed')
    for part, size in (('train', 9527), ('val', 2042), ('test', 2042)):
        x, y = data['x_' + part], data['y_' + part]
        if x.shape != (size, INPUT_DIM) or y.shape != (size,):
            raise ValueError(f'Invalid {part} shapes: {x.shape}, {y.shape}')
        if x.dtype != np.float32 or y.dtype != np.int64:
            raise ValueError(f'Invalid {part} dtypes')
        if not np.isfinite(x).all() or y.min() < 0 or y.max() >= NUM_CLASSES:
            raise ValueError(f'Invalid/nonfinite {part} data')
    ids = np.concatenate([data['indices_' + p] for p in ('train', 'val', 'test')])
    if not np.array_equal(np.sort(ids), np.arange(13611)):
        raise ValueError('Splits overlap or omit source rows')
    return data


def initial_weights(seed):
    """Canonical (input, output) float32 weights shared across frameworks."""
    rng = np.random.default_rng(seed)
    return [(rng.uniform(-np.sqrt(6.0 / n), np.sqrt(6.0 / n), (n, m)).astype(np.float32),
             np.zeros(m, dtype=np.float32)) for n, m in zip(DIMS[:-1], DIMS[1:])]


def weights_hash(weights):
    h = hashlib.sha256()
    for w, b in weights:
        h.update(np.ascontiguousarray(w).tobytes())
        h.update(np.ascontiguousarray(b).tobytes())
    return h.hexdigest()


def numpy_forward(x, weights):
    x = np.asarray(x, dtype=np.float32)
    for i, (w, b) in enumerate(weights):
        x = x @ w + b
        if i < len(weights) - 1:
            x = np.maximum(x, 0)
    return x


def batch_indices(n, epoch, seed):
    order = np.random.default_rng(seed + 10000 + epoch).permutation(n)
    for start in range(0, n, BATCH_SIZE):
        yield order[start:start + BATCH_SIZE]


def batch_order_hash(n, seed):
    h = hashlib.sha256()
    for epoch in range(EPOCHS):
        for ids in batch_indices(n, epoch, seed):
            h.update(np.asarray(ids, dtype='<i8').tobytes())
    return h.hexdigest()


def check_finite(values, context):
    if not all(np.isfinite(np.asarray(v)).all() for v in values):
        raise FloatingPointError(f'Nonfinite loss/gradient/parameter: {context}')


def check_initialization(seed, data, actual_weights, actual_logits):
    expected = initial_weights(seed)
    for (w, b), (aw, ab) in zip(expected, actual_weights, strict=True):
        np.testing.assert_array_equal(w, aw)
        np.testing.assert_array_equal(b, ab)
    reference = numpy_forward(data['x_val'][:32], expected)
    np.testing.assert_allclose(actual_logits, reference, atol=2e-5, rtol=2e-5)
    return float(np.max(np.abs(actual_logits - reference)))


def ce(logits, labels):
    a = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    if a.ndim != 2 or a.shape[1] != NUM_CLASSES or y.shape != (len(a),) or len(a) == 0:
        raise ValueError('Invalid logits/labels shape')
    if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= NUM_CLASSES:
        raise ValueError('Invalid labels')
    check_finite([a], 'cross entropy logits')
    z = a - a.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(y)), y]))


def evaluate(logits, labels):
    cross_entropy = ce(logits, labels)
    pred = np.asarray(logits).argmax(axis=1)
    labels = np.asarray(labels)
    cm = np.bincount(NUM_CLASSES * labels + pred, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    tp = cm.diagonal().astype(float)
    p = np.divide(tp, cm.sum(axis=0), out=np.zeros(NUM_CLASSES), where=cm.sum(axis=0) > 0)
    r = np.divide(tp, cm.sum(axis=1), out=np.zeros(NUM_CLASSES), where=cm.sum(axis=1) > 0)
    f = np.divide(2*p*r, p+r, out=np.zeros(NUM_CLASSES), where=p+r > 0)
    return dict(accuracy=float(np.mean(pred == labels)), precision_macro=float(p.mean()),
                recall_macro=float(r.mean()), f1_macro=float(f.mean()), test_cross_entropy=cross_entropy), cm


def source_fingerprint(framework):
    return {name: file_hash(ROOT / name) for name in ('common.py', FRAMEWORKS[framework])}


def run_path(framework, seed):
    return OUTPUT_DIR / framework / f'seed_{seed}'


def validate_run(framework, seed, data=None):
    """Recompute metrics and verify artifacts before reusing completed runs."""
    data = load_data() if data is None else data
    out = run_path(framework, seed)
    m = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    for key, value in PROTOCOL.items():
        if m.get(key) != value:
            raise ValueError(f'{out}: mismatched {key}')
    if m.get('status') != 'complete' or m['framework'] != framework or m['seed'] != seed:
        raise ValueError(f'{out}: incomplete or incorrect identity')
    if m['version'].split('+')[0] != EXPECTED_VERSIONS[framework]:
        raise ValueError(f'{out}: wrong framework version')
    if m['dataset_sha256'] != file_hash(DATA_PATH) or m['source_files_sha256'] != source_fingerprint(framework):
        raise ValueError(f'{out}: data or source changed')
    if m['initial_weights_sha256'] != weights_hash(initial_weights(seed)):
        raise ValueError(f'{out}: initialization differs')
    if m['batch_order_sha256'] != batch_order_hash(len(data['y_train']), seed):
        raise ValueError(f'{out}: training order differs')
    for name, sha in m['artifacts_sha256'].items():
        if file_hash(out / name) != sha:
            raise ValueError(f'{out}: artifact changed: {name}')
    history = np.loadtxt(out / 'learning_curve.csv', delimiter=',', skiprows=1, ndmin=2)
    if history.shape != (EPOCHS, 3) or not np.array_equal(history[:, 0], np.arange(1, EPOCHS+1)):
        raise ValueError(f'{out}: incomplete learning curve')
    check_finite([history, [m[k] for k in METRIC_FIELDS]], str(out))
    if np.any(history[:, 1:] < 0) or any(m[k] < 0 for k in METRIC_FIELDS):
        raise ValueError(f'{out}: negative loss/metric/time')
    with np.load(out / 'predictions.npz', allow_pickle=False) as p:
        np.testing.assert_array_equal(p['labels'], data['y_test'])
        np.testing.assert_array_equal(p['predicted'], p['logits'].argmax(axis=1))
        np.testing.assert_array_equal(p['source_indices'], data['indices_test'])
        metrics, cm = evaluate(p['logits'], p['labels'])
        with np.load(out / 'final_weights.npz', allow_pickle=False) as weights:
            canonical = [(weights[f'w{i}'], weights[f'b{i}']) for i in range(4)]
            for (n, k), (w, b) in zip(zip(DIMS[:-1], DIMS[1:]), canonical):
                if w.shape != (n, k) or b.shape != (k,):
                    raise ValueError(f'{out}: invalid final layer')
                check_finite([w, b], str(out))
            np.testing.assert_allclose(numpy_forward(data['x_test'], canonical), p['logits'], atol=5e-5, rtol=5e-5)
    for key, value in metrics.items():
        if not np.isclose(value, m[key], atol=1e-12, rtol=0):
            raise ValueError(f'{out}: inconsistent {key}')
    np.testing.assert_array_equal(cm, m['confusion_matrix'])
    np.testing.assert_array_equal(cm, np.loadtxt(out / 'confusion_matrix.csv', delimiter=',', dtype=int))
    if not np.isclose(m['training_plus_validation_seconds'], m['train_seconds'] + m['validation_seconds']):
        raise ValueError(f'{out}: inconsistent timing')
    if m['total_seconds'] < m['training_plus_validation_seconds'] + m['test_seconds']:
        raise ValueError(f'{out}: invalid total time')
    return m


def save_run(framework, version, seed, history, training_seconds, validation_seconds,
             logits, data, final_weights, initial_error, test_seconds, total_seconds, runtime):
    out = run_path(framework, seed)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    stage = out.with_name(out.name + f'.pending_{os.getpid()}')
    stage.mkdir(parents=True)
    metrics, cm = evaluate(logits, data['y_test'])
    hist = np.asarray(history, dtype=float)
    check_finite([hist, [training_seconds, validation_seconds, test_seconds, total_seconds]], framework)
    if hist.shape != (EPOCHS, 3):
        raise ValueError('A successful run must contain all 100 epochs')
    np.savez_compressed(stage / 'predictions.npz', logits=logits, labels=data['y_test'],
                        predicted=np.asarray(logits).argmax(axis=1), source_indices=data['indices_test'])
    np.savez_compressed(stage / 'final_weights.npz',
                        **{f'{name}{i}': arr for i, pair in enumerate(final_weights)
                           for name, arr in zip(('w', 'b'), pair)})
    np.savetxt(stage / 'confusion_matrix.csv', cm, delimiter=',', fmt='%d')
    with (stage / 'learning_curve.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_ce', 'validation_ce'])
        writer.writerows(history)
    artifacts = {p.name: file_hash(p) for p in stage.iterdir() if p.is_file()}
    metrics.update(PROTOCOL)
    metrics.update(status='complete', framework=framework, version=str(version), seed=seed,
                   python_version=platform.python_version(), python_executable=sys.executable,
                   platform=platform.platform(), runtime=runtime,
                   train_seconds=training_seconds, validation_seconds=validation_seconds,
                   training_plus_validation_seconds=training_seconds + validation_seconds,
                   test_seconds=test_seconds, total_seconds=total_seconds,
                   timing_scope='Model setup through final test; excludes imports, loading, saving. Train includes first graph trace, batch conversion and loss synchronization. Validation includes NumPy CE.',
                   dataset_sha256=file_hash(DATA_PATH), source_files_sha256=source_fingerprint(framework),
                   artifacts_sha256=artifacts, initial_weights_sha256=weights_hash(initial_weights(seed)),
                   batch_order_sha256=batch_order_hash(len(data['y_train']), seed),
                   initial_forward_max_abs_error=initial_error,
                   last_train_cross_entropy=float(hist[-1, 1]), last_validation_cross_entropy=float(hist[-1, 2]),
                   confusion_matrix=cm.tolist())
    write_json(stage / 'metrics.json', metrics)
    stage.rename(out)
    validate_run(framework, seed, data)
    print(f'{framework} seed={seed}: accuracy={metrics["accuracy"]:.5f}, '
          f'F1={metrics["f1_macro"]:.5f}, train={training_seconds:.2f}s, val={validation_seconds:.2f}s', flush=True)
    return metrics


def trainer_main(framework, version, run):
    parser = argparse.ArgumentParser(description=f'{framework}: controlled CPU experiments')
    parser.add_argument('--seed', type=int, choices=SEEDS, help='Run one seed; default: all five')
    args = parser.parse_args()
    if str(version).split('+')[0] != EXPECTED_VERSIONS[framework]:
        raise RuntimeError(f'{framework} {version}: expected {EXPECTED_VERSIONS[framework]}')
    data = load_data()
    for seed in (SEEDS if args.seed is None else (args.seed,)):
        if run_path(framework, seed).exists():
            validate_run(framework, seed, data)
            print(f'Skipping verified complete run: {framework} seed={seed}', flush=True)
        else:
            run(seed, data)
