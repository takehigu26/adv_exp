import random
import numpy as np
import tensorflow as tf
import datasets as ds
from evaluate import my_accuracy_score
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
tf.compat.v1.enable_eager_execution()

# set number of threads
import os
import warnings
warnings.filterwarnings('ignore')
#スレッド数を5に設定
os.environ["OMP_NUM_THREADS"] = "10"
from tensorflow.python.eager import context
tf.config.threading.set_intra_op_parallelism_threads(10)
_ = tf.Variable([1])


# build original model
def build_model(n_feature, seed=49, num_layers=3, num_hidden=100, optimizer=None):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=n_feature))
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(num_hidden, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.03)))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

# fitting(called only in original_model training)
def fit_model(model, X_train, y_train, EPOCHS=200, batch_size=256, verbose=0, **kwargs):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
    model.fit(X_train, y_train,
                batch_size = batch_size,
                epochs = EPOCHS,
                callbacks = [lr_decay],
                verbose = verbose)
    return model

'''
def fit_model2(model, X_train, y_train, EPOCHS=200, batch_size=32, verbose=0, **kwargs):
    dataset_train = get_batched_data(X_train, y_train, batch_size)
    min_loss = 999.
    for epoch in range(EPOCHS):
        optimizer = tf.keras.optimizers.Adam(lr=step_decay(epoch))
        for X, y in dataset_train:
            with tf.GradientTape(persistent=True) as t:
                loss = tf.keras.losses.categorical_crossentropy(y, model(X))
            grads = t.gradient(loss, model.trainable_variables)
            del t
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_norm = float(tf.norm(loss, 2))
            if loss_norm < min_loss:
                min_loss = loss_norm
                best_weights = model.get_weights()
    model.set_weights(best_weights)
    return model
'''
def fit_model2(model, X_train, y_train, EPOCHS=200, batch_size=32, verbose=0, **kwargs):
    dataset_train = get_batched_data(X_train, y_train, batch_size)
    for epoch in range(EPOCHS):
        optimizer = tf.keras.optimizers.Adam(lr=step_decay(epoch))
        for X, y in dataset_train:
            with tf.GradientTape(persistent=True) as t:
                loss = tf.keras.losses.categorical_crossentropy(y, model(X))
            grads = t.gradient(loss, model.trainable_variables)
            del t
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return model


# utils
def get_batched_data(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((tf.cast(X, "float32"), tf.cast(y, "float32")))
    return dataset.batch(batch_size)

def step_decay(epoch):
    if epoch < 900: return 0.01
    else: return 0.001
    
lr_decay = LearningRateScheduler(step_decay)

'''
def adv_train(dataset, model, targets, optimizer, verbose, alpha, norm=1, **kwargs):
    loss = tf.keras.losses.categorical_crossentropy
    total_loss_sum = 0
    for X, y in dataset:
        with tf.GradientTape(persistent = True) as t:
            t.watch(X)
            y_pred = model(X)
            perf_loss = loss(y, y_pred)
            exp_loss = tf.Variable(0.)
            exp_loss_entire = t.gradient(perf_loss, X)
            for target in targets:
                exp_loss_r = exp_loss_entire[:, target[0]]
                exp_loss_r = tf.norm(exp_loss_r, norm)
                exp_loss_r = exp_loss_r * target[1]
                exp_loss.assign_add(exp_loss_r)
            total_loss = perf_loss + (alpha * exp_loss / tf.cast(X.shape[0], tf.float32))
            total_loss_sum += tf.norm(total_loss, 1)
            grads = t.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        del t
    return model, total_loss_sum
'''

'''
# adversarial exampleみたいなやつ
def adv_train(dataset, model, targets, optimizer, verbose, alpha, norm=1, **kwargs):
    loss = tf.keras.losses.categorical_crossentropy
    total_loss_sum = 0
    for X, y in dataset:
        delta = no_attack(model, X, y)
        #delta = fgsm(model, X, y)
        with tf.GradientTape(persistent = True) as t:
            t.watch(X)
            t.watch(delta)
            y_pred = model(X + delta)
            perf_loss = loss(y, y_pred)
            l = model.layers[-1]
            l.activation = tf.keras.activations.linear
            exp_loss = tf.Variable(0.)
            exp_loss_entire = t.gradient(perf_loss, X)
            for target in targets:
                exp_loss_r = exp_loss_entire[:, target[0]]
                exp_loss_r = tf.norm(exp_loss_r, norm)
                exp_loss_r = exp_loss_r * target[1]
                exp_loss.assign_add(exp_loss_r)
            total_loss = perf_loss + (alpha * exp_loss / tf.cast(X.shape[0], tf.float32))
            total_loss_sum += tf.norm(total_loss, 1)
            l.activation = tf.keras.activations.softmax
            grads = t.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        del t
    return model, total_loss_sum
'''
def no_attack(model, X, y, **kwargs):
    delta = tf.Variable(tf.zeros_like(X.shape[1:], dtype="float32"))
    return delta


def fgsm(model, X, y, epsilon=0.1, **kwargs):
    """ Construct FGSM adversarial examples on the examples X"""
    y = tf.cast(y, 'float32')
    delta = tf.Variable(tf.zeros_like(X))
    loss = tf.keras.losses.categorical_crossentropy
    loss_eval = loss(model(X + delta), y)
    optimizer = tf.keras.optimizers.RMSprop()
    with tf.GradientTape() as t:
        t.watch(delta)
        loss_eval = loss(model(X + delta), y)
    grads = t.gradient(loss_eval, delta)
    optimizer.apply_gradients(zip(grads, [delta]))
    print(epsilon * tf.sign(grads))
    return epsilon * tf.sign(grads)


def adv_train(dataset, model, targets, optimizer, verbose, alpha, norm=1, **kwargs):
    loss = tf.keras.losses.categorical_crossentropy
    total_loss_sum = 0
    for X, y in dataset:
        delta = no_attack(model, X, y)
        #delta = fgsm(model, X, y)
        with tf.GradientTape(persistent = True) as t:
            t.watch(X)
            t.watch(delta)
            y_pred = model(X + delta)
            perf_loss = loss(y, y_pred)
            l = model.layers[-1]
            l.activation = tf.keras.activations.linear
            exp_loss = tf.Variable(0.)
            for target in targets:
                explanation = t.gradient(perf_loss, [X])[0][:, target[0]]
                exp_loss_r = tf.norm(explanation, norm)
                exp_loss_r = exp_loss_r * target[1]
                exp_loss.assign_add(exp_loss_r)
            total_loss = perf_loss + (alpha * exp_loss / tf.cast(X.shape[0], tf.float32))
            total_loss_sum += tf.norm(total_loss, 1)
            l.activation = tf.keras.activations.softmax
            grads = t.gradient(total_loss, [*model.weights])
            optimizer.apply_gradients(zip(grads, model.weights))
        del t
    return model, total_loss_sum