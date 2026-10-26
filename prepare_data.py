"""Validate the real UCI dataset, split without duplicate leakage, scale TRAIN only."""
from common import ROOT, DATA_PATH, SPLIT_SEED, file_hash, write_json
from pathlib import Path
from io import BytesIO, StringIO
import argparse
import zipfile
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_results import format_number, format_numeric_axes

CLASS_COUNTS = {'BARBUNYA': 1322, 'BOMBAY': 522, 'CALI': 1630, 'DERMASON': 3546,
                'HOROZ': 1928, 'SEKER': 2027, 'SIRA': 2636}
FEATURES = ('Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation',
            'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness',
            'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4')


def read_archive(blob, max_depth=2):
    with zipfile.ZipFile(BytesIO(blob)) as archive:
        names = [n for n in archive.namelist() if not n.endswith('/')]
        for ext in ('.arff', '.xlsx'):
            for name in names:
                if name.lower().endswith(ext):
                    return ext, archive.read(name), name
        if max_depth:
            for name in names:
                if name.lower().endswith('.zip'):
                    try:
                        ext, content, inner = read_archive(archive.read(name), max_depth-1)
                        return ext, content, name + '/' + inner
                    except (ValueError, zipfile.BadZipFile):
                        continue
    raise ValueError('No .arff or .xlsx in UCI archive')


def read_source(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext == '.zip':
        ext, content, member = read_archive(path.read_bytes())
    elif ext in ('.arff', '.xlsx'):
        content, member = path.read_bytes(), path.name
    else:
        raise ValueError('Expected original UCI .zip, .arff or .xlsx')
    if ext == '.arff':
        records, _ = arff.loadarff(StringIO(content.decode('utf-8-sig')))
        frame = pd.DataFrame(records)
        frame['Class'] = frame['Class'].map(lambda v: v.decode() if isinstance(v, bytes) else str(v))
    else:
        frame = pd.read_excel(BytesIO(content))
    frame.columns = [str(c).strip() for c in frame.columns]
    if list(frame.columns) != [*FEATURES, 'Class'] or frame.shape != (13611, 17):
        raise ValueError(f'Unexpected UCI schema: {frame.shape}, {frame.columns.tolist()}')
    if frame.isna().any().any() or frame['Class'].value_counts().to_dict() != CLASS_COUNTS:
        raise ValueError('Missing values or unexpected classes/counts')
    x = frame[list(FEATURES)].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite features')
    return x, frame['Class'].to_numpy(dtype=str), list(FEATURES), member


def stratified_group_indices(x, y):
    """Exact sklearn per-class row targets, keeping duplicate feature rows together."""
    ids = np.arange(len(y))
    train, other = train_test_split(ids, test_size=.30, stratify=y, random_state=SPLIT_SEED)
    val, test = train_test_split(other, test_size=.5, stratify=y[other], random_state=SPLIT_SEED)
    targets = np.array([np.bincount(y[a], minlength=7) for a in (train, val, test)])
    _, groups = np.unique(x, axis=0, return_inverse=True)
    members = {}
    for idx, group in enumerate(groups):
        members.setdefault(int(group), []).append(idx)
    rng = np.random.default_rng(SPLIT_SEED)
    parts = [[], [], []]
    for label in range(7):
        chunks = [np.asarray(v, dtype=np.int64) for v in members.values() if y[v[0]] == label]
        if any(np.any(y[v] != label) for v in chunks):
            raise ValueError('Identical features have conflicting labels')
        rng.shuffle(chunks)
        chunks.sort(key=len, reverse=True)
        remaining = targets[:, label].copy()
        for chunk in chunks:
            available = np.where(remaining >= len(chunk), remaining, 0).astype(float)
            if available.sum() == 0:
                raise ValueError('Cannot satisfy stratified targets with duplicate groups')
            split = int(rng.choice(3, p=available/available.sum()))
            parts[split].extend(chunk.tolist())
            remaining[split] -= len(chunk)
        if remaining.any():
            raise ValueError('Unfilled stratification target')
    result = [rng.permutation(np.asarray(p, dtype=np.int64)) for p in parts]
    sets = [set(groups[p]) for p in result]
    assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i+1, 3))
    return result, groups, targets


def plot_dataset_figures(arrays):
    """Render Ukrainian figures from the existing split without changing model inputs."""
    names = arrays['class_names']
    rows = sum(len(arrays['y_'+part]) for part in ('train', 'val', 'test'))
    scaled = np.empty((rows, 16), dtype=np.float32)
    y = np.empty(rows, dtype=np.int64)
    for part in ('train', 'val', 'test'):
        ids = arrays['indices_'+part]
        scaled[ids] = arrays['x_'+part]
        y[ids] = arrays['y_'+part]
    plots = ROOT / 'figures'; plots.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = [int(np.count_nonzero(y == i)) for i in range(len(names))]
    bars = ax.bar(names, counts, color='#237b91')
    ax.bar_label(bars, padding=3)
    ax.set(xlabel='Клас квасолі', ylabel='Кількість спостережень',
           title='Dry Bean: розподіл класів (13 611 записів)')
    ax.set_ylim(0, max(counts)*1.15); ax.tick_params(axis='x', labelrotation=25)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: format_number(value)))
    fig.tight_layout(); fig.savefig(plots / 'class_distribution.png', dpi=300); plt.close(fig)
    pca = PCA(n_components=2, svd_solver='full').fit(arrays['x_train'])
    scores = pca.transform(scaled)
    ratio = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, name in enumerate(names):
        ax.scatter(*scores[y == i].T, s=5, alpha=.45, label=name, rasterized=True)
    ax.set(xlabel=f'Перша ГК ({format_number(ratio[0]*100, 1)} % дисперсії)',
           ylabel=f'Друга ГК ({format_number(ratio[1]*100, 1)} % дисперсії)',
           title='Dry Bean: проєкція на головні компоненти\nМасштабування й PCA навчені лише на навчальній вибірці')
    format_numeric_axes(ax)
    ax.legend(markerscale=2, fontsize=8); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(plots / 'pca_classes.png', dpi=300); plt.close(fig)
    np.savez_compressed(plots / 'pca_projection.npz', scores=scores, labels=y,
                        components=pca.components_, explained_variance_ratio=ratio)
    return ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', type=Path, default=ROOT / 'Dry_Bean_Dataset.xlsx')
    parser.add_argument('--plots-only', action='store_true', help='Regenerate figures without changing prepared data')
    args = parser.parse_args()
    if args.plots_only:
        from common import load_data
        plot_dataset_figures(load_data())
        print('Dataset figures updated; prepared data unchanged.')
        return
    source = args.source.expanduser().resolve()
    source_sha = file_hash(source)
    if DATA_PATH.exists():
        with np.load(DATA_PATH, allow_pickle=False) as existing:
            if str(existing.get('source_sha256', '')) == source_sha and int(existing.get('split_seed', -1)) == SPLIT_SEED:
                from common import load_data
                load_data()
                print('Prepared data matches the source; reused unchanged.')
                return
        raise FileExistsError('Prepared data differs. Archive it before preparing another split.')
    x, labels, columns, member = read_source(source)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int64)
    parts, groups, targets = stratified_group_indices(x, y)
    scaler = StandardScaler().fit(x[parts[0]])
    arrays = {}
    for part, ids in zip(('train', 'val', 'test'), parts):
        arrays.update({f'x_{part}': scaler.transform(x[ids]).astype(np.float32),
                       f'y_{part}': y[ids], f'indices_{part}': ids, f'groups_{part}': groups[ids]})
    arrays.update(feature_names=np.array(columns), class_names=encoder.classes_,
                  feature_mean=scaler.mean_, feature_scale=scaler.scale_,
                  source_sha256=np.array(source_sha), split_seed=np.array(SPLIT_SEED))
    np.savez_compressed(DATA_PATH, **arrays)
    metadata = dict(source='UCI Dry Bean Dataset (ID 602)',
                    source_url='https://archive.ics.uci.edu/dataset/602/dry+bean+dataset',
                    doi='10.24432/C50S4B', license='CC BY 4.0',
                    original_local_filename=source.name, source_sha256=source_sha, source_member=member,
                    prepared_sha256=file_hash(DATA_PATH), split_seed=SPLIT_SEED,
                    total_rows=len(y), feature_count=16, class_count=7,
                    train_rows=len(parts[0]), validation_rows=len(parts[1]), test_rows=len(parts[2]),
                    class_names=encoder.classes_.tolist(), feature_names=columns,
                    class_counts=CLASS_COUNTS, split_class_counts=targets.tolist(),
                    duplicate_extra_rows=int(len(y)-len(np.unique(groups))),
                    preprocessing='Stratified duplicate-group-aware 70/15/15; exact per-class row targets; StandardScaler fit on TRAIN only')
    ratio = plot_dataset_figures(arrays)
    metadata['pca_train_explained_variance_ratio'] = ratio.tolist()
    write_json(ROOT / 'dataset_metadata.json', metadata)
    print(f'Prepared {len(y)} real rows: train={len(parts[0])}, val={len(parts[1])}, test={len(parts[2])}; no duplicate leakage.')


if __name__ == '__main__':
    main()
