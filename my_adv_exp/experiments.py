from datasets import prep_data
from utils_across import get_original_model, get_modified_model, get_modified_model2
from lime import lime_tabular
import matplotlib.pyplot as plt
import numpy as np
from evaluate import my_accuracy_score
import random

'''
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
    '''
'''
def plot_ranking_histogram(get_dataset, target_index, targets=None, test_index=6, lr=0.01, epochs=5,  **kwargs):
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
    print(">>> average ranking : " + str(sum(rankings) / len(rankings)))
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
        #model = get_modified_model2(get_dataset, targets, verbose=0)
        
        model = get_modified_model(get_dataset, 
                                    targets, 
                                    lr=lr,
                                    alpha=1.0,
                                    batch_size=1000, 
                                    epochs_adv=100,
                                    model_orig=model,
                                    verbose=0)
        
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, _ = prep_data(Xtr, Xts, ytr, yts)
    np.random.seed(1)
    model_original_np = lambda X: model(X).numpy()
    
    #explainer = lime_tabular.LimeTabularExplainer(X_train,
                                                  feature_names = ["feature_"+str(i) for i in range(24)],#df.columns,
                                                  class_names = ['Good', 'Bad'])
    
    explainer = lime_tabular.LimeTabularExplainer(X_train,
                                                  feature_names = ["feature_"+str(i) for i in range(Xtr.shape[1])],#df.columns,
                                                  class_names = ['0', '1'])
    exp = explainer.explain_instance(X_test[test_index], model_original_np, num_features=Xtr.shape[1], top_labels=1)
    for key in exp.local_exp.keys():
        l = list(np.array(exp.local_exp[key], np.int64)[:, 0])
        break

    return l.index(target_index) + 1, my_accuracy_score(yts, model(X_test))



def plot_ranking_histogram2(get_dataset, target_feature, targets=None, epochs=50, **kwargs):
    # model
    model = get_original_model(get_dataset, batch_size=200, verbose=0)
    if targets:
        model = get_modified_model(get_dataset, 
                                    targets, 
                                    lr=0.01,
                                    alpha=0.1,
                                    batch_size=200, 
                                    epochs_adv=50,
                                    model_orig=model,
                                    verbose=0)
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, _ = prep_data(Xtr, Xts, ytr, yts)
    model_np = lambda X: model(X).numpy()
    explainer = lime_tabular.LimeTabularExplainer(X_train,
                                                  feature_names = [i for i in range(Xtr.shape[1])],#df.columns,
                                                  class_names = [0, 1])

    # explanation
    rankings = []                                              
    for _ in range(epochs):
        test_index = random.randint(0, X_test.shape[0]-1)
        exp = explainer.explain_instance(X_test[test_index], model_np, num_features=Xtr.shape[1], top_labels=1)
        for key in exp.local_exp.keys():
            l = list(np.array(exp.local_exp[key], np.int64)[:, 0])
            break
        rankings.append(l.index(target_feature)+1)

    figure_title = "rank frequency of feature_" + str(target_index)
    color = 'mediumseagreen'
    if targets is None:
        figure_title = figure_title + " in original model"
        color = 'royalblue'
    else:
        figure_title = figure_title + " in modified model"
    plt.title(figure_title, fontsize=15)
    plt.xlabel('ranking', fontsize=15)
    plt.ylabel('number of times', fontsize= 15)
    plt.hist(rankings, bins=Xtr.shape[1], range=(1, Xtr.shape[1]), color=color)
    
    print(rankings)
'''


def get_rankings(get_dataset, target_index, targets, epochs=50, lr=0.01, alpha=0.1, batch_size=200, epochs_adv=100, **kwargs):
    rankings_ori = [[] for _ in range(len(targets))]
    rankings_mod = [[] for _ in range(len(targets))]
    accs_ori = []
    accs_mod = []
    accuracies = []
    for epoch in range(epochs):
        ranking_ori, ranking_mod, acc_ori, acc_mod = get_ranking(get_dataset, target_index, targets, 
                                                                 lr, alpha, batch_size, epochs_adv)
        accs_ori.append(acc_ori)
        accs_mod.append(acc_mod)
        for i in range(len(ranking_ori)):
            rankings_ori[i].append(ranking_ori[i])
            rankings_mod[i].append(ranking_mod[i])
        print("#", end="")
        if (epoch+1)%10 == 0: print(str(epoch+1), end="")
    print()
    print("mean accuracy of original model : " + str(sum(accs_ori) / len(accs_ori)))     
    print("mean accuracy of modified model : " + str(sum(accs_mod) / len(accs_mod)))     
    return rankings_ori, rankings_mod    
    
def get_ranking(get_dataset, target_index, targets, lr, alpha, batch_size, epochs_adv):
    ranking_ori, ranking_mod = [], []
    model_original = get_original_model(get_dataset, batch_size=batch_size, verbose=0)
    model_modified = get_modified_model(get_dataset, 
                                        targets, 
                                        lr=lr,
                                        alpha=alpha,
                                        batch_size=batch_size, 
                                        epochs_adv=epochs_adv,
                                        model_orig=model_original,
                                        verbose=0)
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, y_train = prep_data(Xtr, Xts, ytr, yts)
    np.random.seed(1)
    model_original_np = lambda X: model_original(X).numpy()
    model_modified_np = lambda X: model_modified(X).numpy()
    explainer = lime_tabular.LimeTabularExplainer(X_train,
                                                  feature_names = ["feature_"+str(i) for i in range(Xtr.shape[1])],#df.columns,
                                                  class_names = [str(i) for i in range(y_train.shape[1])])
    exp = explainer.explain_instance(X_test[target_index], model_original_np, num_features=Xtr.shape[1], top_labels=1)
    for key in exp.local_exp.keys():
        l_ori = list(np.array(exp.local_exp[key], np.int64)[:, 0])
        break
    exp = explainer.explain_instance(X_test[target_index], model_modified_np, num_features=Xtr.shape[1], top_labels=1)
    for key in exp.local_exp.keys():
        l_mod = list(np.array(exp.local_exp[key], np.int64)[:, 0])
        break
    for target in targets:
        ranking_ori.append(l_ori.index(target[0])+1)
        ranking_mod.append(l_mod.index(target[0])+1)
    return ranking_ori, ranking_mod, my_accuracy_score(yts, model_original(X_test)), my_accuracy_score(yts, model_modified(X_test))

def get_rankings_other(get_dataset, target_index, targets, feature_index, epochs=50, lr=0.01, alpha=0.1, batch_size=200, epochs_adv=100, **kwargs):
    rankings_ori, rankings_mod = [], []
    for epoch in range(epochs):
        ranking_ori, ranking_mod = get_ranking_other(get_dataset, target_index, targets, feature_index,
                                                                 lr, alpha, batch_size, epochs_adv)
        rankings_ori.append(ranking_ori)
        rankings_mod.append(ranking_mod)
        print("#", end="")
        if (epoch+1)%10 == 0: print(str(epoch+1), end="")
    print()   
    return rankings_ori, rankings_mod    
    
def get_ranking_other(get_dataset, target_index, targets, feature_index, lr, alpha, batch_size, epochs_adv):
    ranking_ori, ranking_mod = [], []
    model_original = get_original_model(get_dataset, batch_size=batch_size, verbose=0)
    model_modified = get_modified_model(get_dataset, 
                                        targets, 
                                        lr=lr,
                                        alpha=alpha,
                                        batch_size=batch_size, 
                                        epochs_adv=epochs_adv,
                                        model_orig=model_original,
                                        verbose=0)
    Xtr, Xts, ytr, yts = get_dataset()
    X_test, X_train, _, y_train = prep_data(Xtr, Xts, ytr, yts)
    np.random.seed(1)
    model_original_np = lambda X: model_original(X).numpy()
    model_modified_np = lambda X: model_modified(X).numpy()
    explainer = lime_tabular.LimeTabularExplainer(X_train,
                                                  feature_names = ["feature_"+str(i) for i in range(Xtr.shape[1])],#df.columns,
                                                  class_names = [str(i) for i in range(y_train.shape[1])])
    exp = explainer.explain_instance(X_test[target_index], model_original_np, num_features=Xtr.shape[1], top_labels=1)
    for key in exp.local_exp.keys():
        l_ori = list(np.array(exp.local_exp[key], np.int64)[:, 0])
        break
    exp = explainer.explain_instance(X_test[target_index], model_modified_np, num_features=Xtr.shape[1], top_labels=1)
    for key in exp.local_exp.keys():
        l_mod = list(np.array(exp.local_exp[key], np.int64)[:, 0])
        break
    for target in targets:
        ranking_ori.append(l_ori.index(target[0])+1)
        ranking_mod.append(l_mod.index(target[0])+1)
    return l_ori.index(feature_index) + 1, l_mod.index(feature_index) + 1