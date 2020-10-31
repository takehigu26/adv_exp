from datasets import prep_data
from utils_across import get_original_model, get_modified_model
from lime import lime_tabular
import matplotlib.pyplot as plt
import numpy as np
from evaluate import my_accuracy_score

def plot_ranking_histogram2(get_dataset, model, target_index):
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, _ = prep_data(Xtr, Xts, ytr, yts)
    np.random.seed(1)
    model_np = lambda X: model(X).numpy()
    explainer = lime_tabular.LimeTabularExplainer(X_train,
                                                  feature_names = ["feature_"+str(i) for i in range(24)],#df.columns,
                                                  class_names = ['Good', 'Bad'])
    rankings = []
    for x in X_test:
        exp = explainer.explain_instance(x, model_np, num_features=X_test.shape[-1], top_labels=1)
        return exp.local_exp[1]
        l = list(np.array(exp.local_exp[1], np.int64)[:, 0])
        rankings.append(l.index(target_index) + 1)

    plt.ylim([0, X_test.shape[0]])
    plt.hist(rankings, bins=X_test.shape[1], range=(1, X_test.shape[1]), color='lightseagreen')

def plot_ranking_histogram(get_dataset, target_index, targets=None, test_index=6, lr=0.001, epochs=5,  **kwargs):
    Xtr, _, _, _ = get_dataset()
    n_features = Xtr.shape[-1]
    rankings, accuracies = get_rankings(get_dataset, target_index, targets, test_index, lr, epochs)
    plt.ylim([0, epochs])
    figure_title = "rank frequency of feature_" + str(target_index)
    color = 'mediumseagreen'
    if targets is None:
        figure_title = figure_title + " in original model"
        color = 'royalblue'
    else:
        figure_title = figure_title + " in modified model"
    plt.title(figure_title, fontsize=15)
    plt.xlabel('rank', fontsize=15)
    plt.ylabel('number of times', fontsize= 15)
    plt.hist(rankings, bins=n_features, range=(1, n_features), color=color)
    print(">>> average accuracy : " + str(sum(accuracies) / len(accuracies)))
    #return rankings
    

def get_rankings(get_dataset, target_index, targets, test_index, lr, epochs):
    rankings = []
    accuracies = []
    if targets: print("< modified model >")
    else: print("< original model >")
    for _ in range(epochs):
        ranking, accuracy = get_ranking(get_dataset, target_index, targets, test_index, lr)
        rankings.append(ranking)
        accuracies.append(accuracy)
    return rankings, accuracies
        
        
def get_ranking(get_dataset, target_index, targets, test_index, lr):
    model = get_original_model(get_dataset, batch_size=200, verbose=0)
    if targets:
        model = get_modified_model(get_dataset, 
                                    targets, 
                                    lr=lr,
                                    alpha=0.1,
                                    batch_size=200, 
                                    epochs_adv=50,
                                    model_orig=model,
                                    verbose=0)
        
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, _ = prep_data(Xtr, Xts, ytr, yts)
    np.random.seed(1)
    model_original_np = lambda X: model(X).numpy()
    explainer = lime_tabular.LimeTabularExplainer(X_train,
                                                  feature_names = ["feature_"+str(i) for i in range(24)],#df.columns,
                                                  class_names = ['Good', 'Bad'])
    exp = explainer.explain_instance(X_test[test_index], model_original_np, num_features=24, top_labels=1)
    for key in exp.local_exp.keys():
        l = list(np.array(exp.local_exp[key], np.int64)[:, 0])
        break

    return l.index(target_index) + 1, my_accuracy_score(yts, model(X_test))