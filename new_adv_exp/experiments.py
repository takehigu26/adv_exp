from datasets import prep_data
from train import get_base_model, get_adversarial_model
from lime import lime_tabular
import matplotlib.pyplot as plt
import numpy as np
from evaluate import my_accuracy_score

def plot_ranking_histogram(get_dataset, target_index, targets=None, test_index=6, epochs=5,  **kwargs):
    Xtr, _, _, _ = get_dataset()
    n_features = Xtr.shape[-1]
    rankings, accuracies = get_rankings(get_dataset, target_index, targets=targets, test_index=test_index, epochs=epochs)
    plt.ylim([0, epochs])
    figure_title = "rank frequency of feature_" + str(target_index)
    color = 'mediumseagreen'
    if not targets:
        figure_title = figure_title + " in original model"
        color = 'royalblue'
    else:
        figure_title = figure_title + " in modified model"
    plt.title(figure_title, fontsize=15)
    plt.xlabel('rank', fontsize=15)
    plt.ylabel('number of times', fontsize= 15)
    plt.hist(rankings, bins=n_features, range=(1, n_features), color=color)
    print(">>> average accuracy : " + str(sum(accuracies) / len(accuracies)))
    return rankings
    

def get_rankings(get_dataset, target_index, targets, test_index, epochs=5, **kwargs):
    rankings = []
    accuracies = []
    if targets: print("< modified model >")
    else: print("< original model >")
    for _ in range(epochs):
        ranking, accuracy = get_ranking(get_dataset, target_index, targets=targets, test_index=test_index)
        rankings.append(ranking)
        accuracies.append(accuracy)
    return rankings, accuracies
        
        
def get_ranking(get_dataset, target_index, targets, test_index):
    if targets:
        model = get_adversarial_model(get_dataset)
    else:
        model = get_base_model(get_dataset)
        
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, _ = prep_data(Xtr, Xts, ytr, yts)
    np.random.seed(1)
    model_original_np = lambda X: model(X).numpy()
    explainer = lime_tabular.LimeTabularExplainer(X_train,
                                                  feature_names = ["feature_"+str(i) for i in range(24)],#df.columns,
                                                  class_names = ['Good', 'Bad'])
    exp = explainer.explain_instance(X_test[test_index], model_original_np, num_features=24, top_labels=1)
    l = list(np.array(exp.local_exp[1], np.int64)[:, 0])

    return l.index(target_index) + 1, my_accuracy_score(yts, model(X_test))
