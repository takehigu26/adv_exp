import tensorflow as tf
from tensorflow import keras
import datasets as ds
from evaluate import my_accuracy_score
#from evaluate import feature_importance_nulify, evaluate_pertrubed_models
from train_utils import build_model, fit_model, get_batched_data, adv_train, fit_model2, step_decay

# set number of threads
import os
import warnings
warnings.filterwarnings('ignore')
#スレッド数を5に設定
os.environ["OMP_NUM_THREADS"] = "10"
from tensorflow.python.eager import context
tf.config.threading.set_intra_op_parallelism_threads(10)
_ = tf.Variable([1])


def get_original_model(get_dataset, optimizer=None,seed=49, num_layers=3, batch_size=200, verbose=1, **kwargs):
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, y_train = ds.prep_data(Xtr, Xts, ytr, yts)
    if optimizer is None:   optimizer = tf.keras.optimizers.Adam(lr=0.01)
    EPOCHS = 1000
    model_notlearned = build_model(Xtr.shape[-1], num_layers=num_layers, optimizer=optimizer, seed=seed)
    if verbose: print("accuracy_score(not learnt yet) : " + str(my_accuracy_score(yts, model_notlearned(X_test))))
    history = fit_model(model_notlearned, X_train, y_train, EPOCHS, batch_size=batch_size, verbose=0)
    #history = fit_model2(model_notlearned, X_train, y_train, EPOCHS, batch_size=batch_size, verbose=0)
    accuracy = my_accuracy_score(yts, history(X_test))
    if verbose: print(">>> accuracy_score(original_model) : " + str(accuracy))
    return history#, accuracy
'''
def get_modified_model(get_dataset, targets, optimizer=None, epochs_adv=50, verbose=1, alpha=3.0, model_orig=None, seed=49, num_layers=3, batch_size=200, **kwargs):
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, y_train = ds.prep_data(Xtr, Xts, ytr, yts)
    dataset_train = get_batched_data(X_train, y_train, batch_size)

    # build base-model
    if optimizer is None: optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model_modified = build_model(Xtr.shape[-1], num_layers=num_layers, optimizer=optimizer, seed=seed)

    if model_orig is None:
        model_orig = get_original_model(get_dataset, optimizer=optimizer, batch_size=batch_size, num_layers=num_layers, verbose=0)
    model_modified.set_weights(model_orig.get_weights())
    if verbose: print("accuracy_score(base model) :" + str(my_accuracy_score(yts, model_modified(X_test))))

    # adversarial training
    for i in range(epochs_adv):
        model_modified = adv_train(dataset_train, model_modified, targets, verbose=(verbose-1), alpha=alpha, optimizer=tf.keras.optimizers.Adam(lr=0.001))
        #f (verbose-1): print("score(" + str(i) + ") : " +  str(my_accuracy_score(yts, model_modified(X_test))))

    if verbose: print(">>> accuracy_score(modified_model) : " + str(my_accuracy_score(yts, model_modified(X_test))))
    return model_modified
'''
def get_modified_model(get_dataset, targets, lr=0.01, epochs_adv=50, verbose=1, alpha=0.1, model_orig=None, seed=49, num_layers=3, batch_size=200, **kwargs):
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, y_train = ds.prep_data(Xtr, Xts, ytr, yts)
    dataset_train = get_batched_data(X_train, y_train, batch_size)

    # build base-model
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model_modified = build_model(Xtr.shape[-1], num_layers=num_layers, optimizer=optimizer, seed=seed)
    best_model = build_model(Xtr.shape[-1], num_layers=num_layers, optimizer=optimizer, seed=seed)
    if verbose: print("accuracy_score(haven't learnt) :" + str(my_accuracy_score(yts, model_modified(X_test))))

    if model_orig is None:
        model_orig = get_original_model(get_dataset, optimizer=optimizer, batch_size=batch_size, num_layers=num_layers, verbose=0)
    model_modified.set_weights(model_orig.get_weights())

    min_loss = 9999999999.

    # adversarial training
    for i in range(epochs_adv):
        model_modified, total_loss_sum = adv_train(dataset_train, model_modified, targets, verbose=0, alpha=alpha, optimizer=tf.keras.optimizers.Adam(lr=0.001))
        #print("total_loss: " + str(total_loss_sum))
        if total_loss_sum < min_loss:
            min_loss = total_loss_sum
            best_weights = model_modified.get_weights()
        #print(str(i) + " : " + str(total_loss))
    best_model.set_weights(best_weights)
    print("min_loss : " + str(min_loss))
    if verbose: print(">>> accuracy_score(modified_model) : " + str(my_accuracy_score(yts, best_model(X_test))))
    return best_model

# this uses adversarial loss function since the beginning of trainings
def get_modified_model2(get_dataset, targets, epochs_adv=1000, verbose=1, alpha=0.1, num_layers=3, batch_size=200, **kwargs):
    # datasets
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, y_train = ds.prep_data(Xtr, Xts, ytr, yts)
    dataset_train = get_batched_data(X_train, y_train, batch_size)
    
    # base model
    adv_model = build_model(Xtr.shape[-1], num_layers=num_layers, optimizer=keras.optimizers.Adam(lr=0.01), seed=49)

    # adversarial training
    min_loss = 9999999999.
    for epoch in range(epochs_adv):
        adv_model, total_loss_sum = adv_train(dataset_train, 
                                              adv_model, 
                                              targets, 
                                              verbose=0, 
                                              alpha=alpha, 
                                              optimizer=keras.optimizers.Adam(lr=step_decay(epoch)))
        if total_loss_sum < min_loss:
            min_loss = total_loss_sum
            best_weights = adv_model.get_weights()
    adv_model.set_weights(best_weights)
    print("min_loss : " + str(min_loss))
    if verbose: print(">>> accuracy_score(modified_model) : " + str(my_accuracy_score(yts, adv_model(X_test))))
    return adv_model