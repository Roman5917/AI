"""PaddlePaddle 3.3.0, CPU and native Adam with global norm clipping."""
import common as c
import time
import numpy as np
import paddle


def run(seed, data):
    start = time.perf_counter()
    paddle.seed(seed)
    cells = []
    for i, (n, m) in enumerate(zip(c.DIMS[:-1], c.DIMS[1:])):
        cells.append(paddle.nn.Linear(n, m))
        if i < 3:
            cells.append(paddle.nn.ReLU())
    model = paddle.nn.Sequential(*cells)
    dense = [layer for layer in model.sublayers() if isinstance(layer, paddle.nn.Linear)]
    for layer, (w, b) in zip(dense, c.initial_weights(seed)):
        layer.weight.set_value(w); layer.bias.set_value(b)
    def weights():
        return [(layer.weight.numpy().copy(), layer.bias.numpy().copy()) for layer in dense]
    with paddle.no_grad():
        initial = model(paddle.to_tensor(data['x_val'][:32])).numpy()
    error = c.check_initialization(seed, data, weights(), initial)
    criterion = paddle.nn.CrossEntropyLoss()
    opt = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=c.LEARNING_RATE,
                                beta1=c.BETA1, beta2=c.BETA2, epsilon=c.EPSILON,
                                weight_decay=c.WEIGHT_DECAY, amsgrad=False,
                                grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=c.CLIP_NORM))
    training = validation = 0.
    history = []
    for epoch in range(c.EPOCHS):
        t = time.perf_counter(); model.train(); losses = 0.
        for batch, ids in enumerate(c.batch_indices(len(data['x_train']), epoch, seed), 1):
            x = paddle.to_tensor(data['x_train'][ids])
            y = paddle.to_tensor(data['y_train'][ids], dtype='int64')
            loss = criterion(model(x), y)
            loss.backward(); opt.step(); opt.clear_grad()
            number = float(loss.numpy().item())
            if not np.isfinite(number):
                raise FloatingPointError(f'PaddlePaddle seed={seed} epoch={epoch+1} batch={batch}')
            losses += number * len(ids)
        training += time.perf_counter() - t
        c.check_finite([a for pair in weights() for a in pair], f'PaddlePaddle epoch {epoch+1}')
        t = time.perf_counter(); model.eval()
        with paddle.no_grad():
            logits = model(paddle.to_tensor(data['x_val'])).numpy()
        val = c.ce(logits, data['y_val']); validation += time.perf_counter() - t
        history.append((epoch+1, losses/len(data['x_train']), val))
        if (epoch+1) % 25 == 0:
            print(f'PaddlePaddle seed={seed} epoch={epoch+1}: train_ce={history[-1][1]:.5f} val_ce={val:.5f}', flush=True)
    t = time.perf_counter()
    with paddle.no_grad():
        test = model(paddle.to_tensor(data['x_test'])).numpy()
    test_seconds = time.perf_counter() - t
    total = time.perf_counter() - start
    c.save_run('PaddlePaddle', paddle.__version__, seed, history, training, validation, test, data,
               weights(), error, test_seconds, total,
               dict(mode='dynamic', intra_op_threads=c.THREADS,
                    thread_flags=paddle.get_flags(['FLAGS_paddle_num_threads'])))


if __name__ == '__main__':
    paddle.set_device('cpu')
    paddle.set_flags({'FLAGS_paddle_num_threads': c.THREADS})
    c.trainer_main('PaddlePaddle', paddle.__version__, run)
