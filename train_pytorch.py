"""PyTorch 2.9.1, CPU, native Adam and shared NumPy initialization/order."""
import common as c  # Configure numeric runtimes BEFORE importing torch.
import time
import numpy as np
import torch
from torch import nn


def run(seed, data):
    start = time.perf_counter()
    torch.manual_seed(seed)
    cells = []
    for i, (n, m) in enumerate(zip(c.DIMS[:-1], c.DIMS[1:])):
        cells.append(nn.Linear(n, m))
        if i < 3:
            cells.append(nn.ReLU())
    model = nn.Sequential(*cells)
    dense = [layer for layer in model if isinstance(layer, nn.Linear)]
    with torch.no_grad():
        for layer, (w, b) in zip(dense, c.initial_weights(seed)):
            layer.weight.copy_(torch.from_numpy(w.T.copy()))
            layer.bias.copy_(torch.from_numpy(b))
    def weights():
        return [(layer.weight.detach().numpy().T.copy(), layer.bias.detach().numpy().copy()) for layer in dense]
    with torch.no_grad():
        initial = model(torch.from_numpy(data['x_val'][:32])).numpy()
    error = c.check_initialization(seed, data, weights(), initial)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=c.LEARNING_RATE, betas=(c.BETA1, c.BETA2),
                           eps=c.EPSILON, weight_decay=c.WEIGHT_DECAY, amsgrad=False, foreach=False)
    xt, yt = torch.from_numpy(data['x_train']), torch.from_numpy(data['y_train'])
    xv, xs = torch.from_numpy(data['x_val']), torch.from_numpy(data['x_test'])
    training = validation = 0.
    history = []
    for epoch in range(c.EPOCHS):
        t = time.perf_counter(); model.train(); losses = 0.
        for batch, ids in enumerate(c.batch_indices(len(xt), epoch, seed), 1):
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xt[ids]), yt[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), c.CLIP_NORM, error_if_nonfinite=True)
            opt.step()
            number = float(loss.item())
            if not np.isfinite(number):
                raise FloatingPointError(f'PyTorch seed={seed} epoch={epoch+1} batch={batch}')
            losses += number * len(ids)
        training += time.perf_counter() - t
        c.check_finite([a for pair in weights() for a in pair], f'PyTorch epoch {epoch+1}')
        t = time.perf_counter(); model.eval()
        with torch.no_grad():
            logits = model(xv).numpy()
        val = c.ce(logits, data['y_val']); validation += time.perf_counter() - t
        history.append((epoch+1, losses/len(xt), val))
        if (epoch+1) % 25 == 0:
            print(f'PyTorch seed={seed} epoch={epoch+1}: train_ce={history[-1][1]:.5f} val_ce={val:.5f}', flush=True)
    t = time.perf_counter()
    with torch.no_grad():
        test = model(xs).numpy()
    test_seconds = time.perf_counter() - t
    total = time.perf_counter() - start
    c.save_run('PyTorch', torch.__version__, seed, history, training, validation, test, data,
               weights(), error, test_seconds, total,
               dict(mode='eager', intra_op_threads=torch.get_num_threads(), inter_op_threads=torch.get_num_interop_threads()))


if __name__ == '__main__':
    torch.set_num_threads(c.THREADS)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    c.trainer_main('PyTorch', torch.__version__, run)
