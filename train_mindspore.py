"""MindSpore 2.10.0, CPU PyNative; fail on nonfinite values, never replace them."""
import common as c
import time
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops


def run(seed, data):
    start = time.perf_counter()
    ms.set_seed(seed)
    cells = []
    for i, (n, m) in enumerate(zip(c.DIMS[:-1], c.DIMS[1:])):
        cells.append(nn.Dense(n, m))
        if i < 3:
            cells.append(nn.ReLU())
    model = nn.SequentialCell(cells)
    dense = [layer for layer in cells if isinstance(layer, nn.Dense)]
    for layer, (w, b) in zip(dense, c.initial_weights(seed)):
        layer.weight.set_data(Tensor(w.T.copy(), ms.float32))
        layer.bias.set_data(Tensor(b, ms.float32))
    def weights():
        return [(layer.weight.asnumpy().T.copy(), layer.bias.asnumpy().copy()) for layer in dense]
    initial = model(Tensor(data['x_val'][:32], ms.float32)).asnumpy()
    error = c.check_initialization(seed, data, weights(), initial)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Adam(model.trainable_params(), learning_rate=c.LEARNING_RATE, beta1=c.BETA1,
                  beta2=c.BETA2, eps=c.EPSILON, weight_decay=c.WEIGHT_DECAY, use_nesterov=False)
    def forward(x, y):
        return criterion(model(x), y)
    grad = ms.value_and_grad(forward, None, opt.parameters)
    training = validation = 0.
    history = []
    for epoch in range(c.EPOCHS):
        t = time.perf_counter(); model.set_train(True); losses = 0.
        for batch, ids in enumerate(c.batch_indices(len(data['x_train']), epoch, seed), 1):
            x = Tensor(data['x_train'][ids], ms.float32)
            y = Tensor(data['y_train'][ids].astype(np.int32), ms.int32)
            loss, grads = grad(x, y)
            grads = ops.clip_by_global_norm(grads, clip_norm=c.CLIP_NORM)
            opt(grads)
            number = float(loss.asnumpy())
            if not np.isfinite(number):
                raise FloatingPointError(f'MindSpore seed={seed} epoch={epoch+1} batch={batch}')
            losses += number * len(ids)
        training += time.perf_counter() - t
        c.check_finite([a for pair in weights() for a in pair], f'MindSpore epoch {epoch+1}')
        c.check_finite([g.asnumpy() for g in grads], f'MindSpore gradients epoch {epoch+1}')
        t = time.perf_counter(); model.set_train(False)
        logits = model(Tensor(data['x_val'], ms.float32)).asnumpy()
        val = c.ce(logits, data['y_val']); validation += time.perf_counter() - t
        history.append((epoch+1, losses/len(data['x_train']), val))
        if (epoch+1) % 25 == 0:
            print(f'MindSpore seed={seed} epoch={epoch+1}: train_ce={history[-1][1]:.5f} val_ce={val:.5f}', flush=True)
    t = time.perf_counter()
    test = model(Tensor(data['x_test'], ms.float32)).asnumpy()
    test_seconds = time.perf_counter() - t
    total = time.perf_counter() - start
    c.save_run('MindSpore', ms.__version__, seed, history, training, validation, test, data,
               weights(), error, test_seconds, total,
               dict(mode='PYNATIVE_MODE', intra_op_threads=ms.get_context('runtime_num_threads'),
                    inter_op_threads=ms.get_context('inter_op_parallel_num')))


if __name__ == '__main__':
    ms.set_context(mode=ms.PYNATIVE_MODE, runtime_num_threads=c.THREADS, inter_op_parallel_num=1)
    ms.set_device('CPU')
    c.trainer_main('MindSpore', ms.__version__, run)
