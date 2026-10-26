"""TensorFlow 2.21.0, CPU, native Keras Adam and one traced mini-batch step."""
import common as c
import time
import numpy as np
import tensorflow as tf


def run(seed, data):
    start = time.perf_counter()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(c.INPUT_DIM,)),
                                *[tf.keras.layers.Dense(n, activation='relu') for n in c.HIDDEN],
                                tf.keras.layers.Dense(c.NUM_CLASSES)])
    dense = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    for layer, (w, b) in zip(dense, c.initial_weights(seed)):
        layer.set_weights([w, b])
    def weights():
        return [tuple(layer.get_weights()) for layer in dense]
    initial = model(data['x_val'][:32], training=False).numpy()
    error = c.check_initialization(seed, data, weights(), initial)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=c.LEARNING_RATE, beta_1=c.BETA1, beta_2=c.BETA2,
                                  epsilon=c.EPSILON, weight_decay=c.WEIGHT_DECAY, amsgrad=False)

    @tf.function(input_signature=[tf.TensorSpec((None, c.INPUT_DIM), tf.float32),
                                  tf.TensorSpec((None,), tf.int64)])
    def step(x, y):
        with tf.GradientTape() as tape:
            loss = criterion(y, model(x, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        grads, norm = tf.clip_by_global_norm(grads, c.CLIP_NORM)
        with tf.control_dependencies([tf.debugging.check_numerics(norm, 'Nonfinite gradient norm')]):
            opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    training = validation = 0.
    history = []
    for epoch in range(c.EPOCHS):
        t = time.perf_counter(); losses = 0.
        for batch, ids in enumerate(c.batch_indices(len(data['x_train']), epoch, seed), 1):
            number = float(step(tf.convert_to_tensor(data['x_train'][ids]),
                                tf.convert_to_tensor(data['y_train'][ids])).numpy())
            if not np.isfinite(number):
                raise FloatingPointError(f'TensorFlow seed={seed} epoch={epoch+1} batch={batch}')
            losses += number * len(ids)
        training += time.perf_counter() - t
        c.check_finite([a for pair in weights() for a in pair], f'TensorFlow epoch {epoch+1}')
        t = time.perf_counter()
        val = c.ce(model(data['x_val'], training=False).numpy(), data['y_val'])
        validation += time.perf_counter() - t
        history.append((epoch+1, losses/len(data['x_train']), val))
        if (epoch+1) % 25 == 0:
            print(f'TensorFlow seed={seed} epoch={epoch+1}: train_ce={history[-1][1]:.5f} val_ce={val:.5f}', flush=True)
    t = time.perf_counter()
    test = model(data['x_test'], training=False).numpy()
    test_seconds = time.perf_counter() - t
    total = time.perf_counter() - start
    c.save_run('TensorFlow', tf.__version__, seed, history, training, validation, test, data,
               weights(), error, test_seconds, total,
               dict(mode='tf.function training; eager validation',
                    intra_op_threads=tf.config.threading.get_intra_op_parallelism_threads(),
                    inter_op_threads=tf.config.threading.get_inter_op_parallelism_threads(),
                    adam_epsilon_note='Keras epsilon-hat convention differs from PyTorch bias-corrected denominator'))


if __name__ == '__main__':
    tf.config.threading.set_intra_op_parallelism_threads(c.THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], 'GPU')
    tf.config.experimental.enable_op_determinism()
    c.trainer_main('TensorFlow', tf.__version__, run)
