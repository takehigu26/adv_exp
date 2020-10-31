from datasets import prep_data
from utils_across import get_original_model, get_modified_model
from lime import lime_tabular
import matplotlib.pyplot as plt
import numpy as np
from evaluate import my_accuracy_score

def plot_ranking_histogram(get_dataset, target_index, targets=None, test_index=6, epochs=5,  **kwargs):
    Xtr, _, _, _ = get_dataset()
    n_features = Xtr.shape[-1]
    rankings_ori, rankings_mod, accuracies_ori, accuracies_mod = get_rankings(get_dataset, target_index, targets=targets, test_index=test_index, epochs=epochs)
    plt.ylim([0, epochs])
    # histograms of original model
    print("original model's histograms")
    print(">> average accuracy: " + str(sum(accuracies_ori)/len(accuracies_ori)))
    for i in range(len(target_index)):
        figure_title = "rank frequency of feature_" + str(target_index[i]) + " in original model"
        color = 'royalblue'
        plt.title(figure_title, fontsize=15)
        plt.xlabel('rank', fontsize=15)
        plt.ylabel('number of times', fontsize= 15)
        plt.hist(rankings_ori[i], bins=n_features, range=(1, n_features), color=color)

    # histograms of modified model
    print("modified model's histograms")
    print(">> average accuracy: " + str(sum(accuracies_mod)/len(accuracies_mod)))
    for i in range(len(target_index)):
        figure_title = "rank frequency of feature_" + str(target_index[i]) + " in modified model"
        color = 'mediumseagreen'
        plt.title(figure_title, fontsize=15)
        plt.xlabel('rank', fontsize=15)
        plt.ylabel('number of times', fontsize= 15)
        plt.hist(rankings_mod[i], bins=n_features, range=(1, n_features), color=color)

def get_rankings(get_dataset, target_index, targets, test_index, epochs=5, **kwargs):
    rankings_ori, rankings_mod = [[] for _ in range(len(target_index))], [[] for _ in range(len(target_index))]
    accuracies_ori, accuracies_mod = [], []
    for _ in range(epochs):
        ranking_ori, ranking_mod, accuracy_ori, accuracy_mod = get_ranking(get_dataset, target_index, targets=targets, test_index=test_index)
        for i in range(len(ranking_ori)):
            rankings_ori[i].append(ranking_ori[i])
            rankings_mod[i].append(ranking_mod[i])
        accuracies_ori.append(accuracy_ori)
        accuracies_mod.append(accuracy_mod)
    return rankings_ori, rankings_mod, accuracies_ori, accuracies_mod

def get_ranking(get_dataset, target_index, targets, test_index):
    model_ori = get_original_model(get_dataset, batch_size=200, verbose=0)
    model_mod = get_modified_model(get_dataset, 
                                    targets, 
                                    batch_size=200, 
                                    epochs_adv=50,
                                    model_orig=model_ori,
                                    verbose=0)
        
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, _ = prep_data(Xtr, Xts, ytr, yts)
    np.random.seed(1)
    model_ori_np = lambda X: model_ori(X).numpy()
    model_mod_np = lambda X: model_mod(X).numpy()
    explainer = lime_tabular.LimeTabularExplainer(X_train,
                                                  feature_names = ["feature_"+str(i) for i in range(24)],#df.columns,
                                                  class_names = ['Good', 'Bad'])
    exp_ori = explainer.explain_instance(X_test[test_index], model_ori_np, num_features=24, top_labels=1)
    exp_mod = explainer.explain_instance(X_test[test_index], model_mod_np, num_features=24, top_labels=1)
    l_ori = list(np.array(exp_ori.local_exp[1], np.int64)[:, 0])
    l_mod = list(np.array(exp_mod.local_exp[1], np.int64)[:, 0])
    ranking_ori = [l_ori.index(idx)+1 for idx in target_index]
    ranking_mod = [l_mod.index(idx)+1 for idx in target_index]
    return ranking_ori, ranking_mod, my_accuracy_score(yts, model_ori(X_test)), my_accuracy_score(yts, model_mod(X_test))