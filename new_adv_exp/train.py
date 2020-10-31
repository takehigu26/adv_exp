import tensorflow as tf
from tensorflow import keras
from datasets import get_german, prep_data
from evaluate import my_accuracy_score
from models import BaseModel, AdversarialModel
from tensorflow.keras.callbacks import LearningRateScheduler
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

# set number of threads
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "10"
from tensorflow.python.eager import context
tf.config.threading.set_intra_op_parallelism_threads(10)
_ = tf.Variable([1])




def get_base_model(get_dataset, seed=49, num_layers=3, batch_size=200, num_layers=3, verbose=1, **kwargs):
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, y_train = prep_data(Xtr, Xts, ytr, yts)

    # layers
    inputs = tf.keras.layers.Input(shape=Xtr.shape[1])
    num_hidden = 100
    layers = [inputs]
    for i in range(1, num_layers+1):
        layers.append(tf.keras.layers.Dense(num_hidden, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.03))(layers[i-1]))
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(layers[num_layers])

    base_model = BaseModel(inputs, outputs)
    #base_model = BaseModel(X_train.shape[-1])
    base_model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                        loss=keras.losses.categorical_crossentropy,
                        metrics=['accuracy'])
    base_model.fit(X_train, y_train,
                    batch_size=200,
                    epochs=1000,
                    callbacks = [lr_decay],
                    verbose=0)
    if verbose: print(">>> accuracy_score(base_model) : " + str(my_accuracy_score(yts, base_model(X_test))))
    return base_model




def get_adversarial_model(get_dataset, targets=None, alpha=0.1, batch_size = 200, num_layers=3, epochs=1000, verbose=1, **kwargs):
    if targets is None: print("targets required")
    
    # dataset
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, y_train = prep_data(Xtr, Xts, ytr, yts)
    X_train, y_train = tf.cast(X_train, tf.float32), tf.cast(y_train, tf.float32)

    # layers
    inputs = tf.keras.layers.Input(shape=Xtr.shape[1])
    num_hidden = 100
    layers = [inputs]
    for i in range(1, num_layers+1):
        layers.append(tf.keras.layers.Dense(num_hidden, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.03))(layers[i-1]))
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(layers[num_layers])

    # model training
    adv_model = AdversarialModel(targets, alpha, inputs, outputs)
    adv_model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                        loss=keras.losses.categorical_crossentropy,
                        metrics=['accuracy'])
    adv_model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks = [lr_decay],
                    verbose=0)
    if verbose: print(">>> accuracy_score(adversarial_model) : " + str(my_accuracy_score(yts, adv_model(X_test))))
    return adv_model


def get_adversarial_model(get_dataset, targets=None, alpha=0.1, batch_size = 200, num_layers=3, epochs=1000, verbose=1, **kwargs):
    if targets is None: print("targets required")
    
    # dataset
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, y_train = prep_data(Xtr, Xts, ytr, yts)
    X_train, y_train = tf.cast(X_train, tf.float32), tf.cast(y_train, tf.float32)

    # layers
    inputs = tf.keras.layers.Input(shape=Xtr.shape[1])
    num_hidden = 100
    layers = [inputs]
    for i in range(1, num_layers+1):
        layers.append(tf.keras.layers.Dense(num_hidden, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.03))(layers[i-1]))
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(layers[num_layers])

    # model training
    adv_model = BaseModel(inputs, outputs)
    adv_model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                        loss=keras.losses.categorical_crossentropy,
                        metrics=['accuracy'])
    adv_model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks = [lr_decay],
                    verbose=0)
    if verbose: print(">>> accuracy_score(adversarial_model) : " + str(my_accuracy_score(yts, adv_model(X_test))))
    return adv_model


# utils
def step_decay(epoch):
    if epoch < 900: return 0.01
    else: return 0.001
lr_decay = LearningRateScheduler(step_decay)