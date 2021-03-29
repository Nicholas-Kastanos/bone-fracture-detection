import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, plot_confusion_matrix)
from tqdm import tqdm
from datetime import datetime

from dataset import XRAYTYPE, get_image_generator
from processing import load_ds

from gp import bayesian_optimisation
from plotters import plot_iteration
from statistics import mean


def _create_SVM_estimator(C, random_state):
    return svm.SVC(C=C, kernel='rbf', random_state=random_state)
def _create_MLP_estimator(input_size, n_hidden, random_state, output_size=2):
    hidden_layers = tuple(np.linspace(input_size, output_size, int(n_hidden)+1, endpoint=False, dtype=np.int)[1:])
    return MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=1000,
            early_stopping=True,
            activation='relu',
            solver='adam',
            random_state=random_state
        )
def _create_RF_estimator(n_estimators, random_state):
    return RandomForestClassifier(n_estimators=int(n_estimators), random_state=random_state)


def sample_loss(params, score='valid', classifier='svm', eval_fn=f1_score, random_state=None, show_confusion=False):
    n_pca_components=int(params[0])
    n_SIFT_features=int(params[1])

    xr_type = XRAYTYPE.FOREARM
    x_train, y_train, x_val, y_val = load_ds(xr_type, n_pca=n_pca_components, n_SIFT_features=n_SIFT_features)

    if classifier=='svm':
        C = 2**params[2]
        clf = _create_SVM_estimator(C, random_state)
    elif classifier=='mlp':
        input_size = x_train.shape[1]
        n_hidden = params[2]
        clf = _create_MLP_estimator(input_size, n_hidden, random_state)
    elif classifier=='rf':
        n_estimators = params[2]
        clf = _create_RF_estimator(n_estimators, random_state)
    elif classifier=='vote':
        input_size = x_train.shape[1]
        C = 2**params[2]
        n_hidden = params[3]
        n_estimators=params[4]
        estimators = [
            ('svm', _create_SVM_estimator(C=C, random_state=random_state)),
            ('mlp', _create_MLP_estimator(input_size=input_size, n_hidden=n_hidden, random_state=random_state)),
            ('rf', _create_RF_estimator(n_estimators=n_estimators, random_state=random_state))
        ]
        clf = VotingClassifier(estimators=estimators, n_jobs=-1)

    elif classifier=='bagging-mlp':
        input_size = x_train.shape[1]
        n_hidden = params[2]
        clf = BaggingClassifier(
            base_estimator=_create_MLP_estimator(input_size=input_size, n_hidden=n_hidden, random_state=None),
            n_estimators=101, 
            n_jobs=-1,
            random_state=random_state
        )

    mean = np.mean(x_train, axis=0)
    stdev = np.std(x_train, axis=0)

    x_train=(x_train-mean)/stdev
    x_val=(x_val-mean)/stdev

    clf.fit(x_train, y_train)

    if score=='train':
        y_pred = clf.predict(x_train)
        if show_confusion:
            # abnormalities == positives == 1
            plot_confusion_matrix(clf, x_train, y_train, display_labels=['Normal', 'Abnormal'], normalize='true', colorbar=False)
            plt.tight_layout()
            plt.show()
        return eval_fn(y_train, y_pred)
    else:
        y_pred = clf.predict(x_val)
        if show_confusion:
            # abnormalities == positives == 1
            plot_confusion_matrix(clf, x_val, y_val, display_labels=['Normal', 'Abnormal'], normalize='true', colorbar=False)
            plt.tight_layout()
            plt.show()
        return eval_fn(y_val, y_pred)

def _get_common_params():
    p_start = 10
    p_end = 10
    p_step = 10
    p_num = ((p_end - p_start)/(p_step)) + 1
    assert np.mod(p_num, 1) == 0
    p_num = int(p_num)

    s_start = 1050
    s_end= 1200
    s_step = 50
    s_num=((s_end - s_start)/(s_step)) + 1
    assert np.mod(s_num, 1) == 0
    s_num = int(s_num)

    pcaFeatures = np.linspace(p_start, p_end, p_num, True, dtype=np.int)
    siftFeatures = np.linspace(s_start, s_end, s_num, True, dtype=np.int)

    pcaBounds=[p_start, p_end]
    siftBounds=[s_start, s_end]

    return pcaFeatures, pcaBounds, siftFeatures, siftBounds

def param_search(param_grid, bounds, classifier, num_loops=10):
    real_loss = []
    print("Running Parameter Grid")
    for params in tqdm(param_grid):
        mean_loss = mean([sample_loss(params, classifier=classifier) for _ in range(num_loops)])
        real_loss.append(mean_loss)

    # The maximum is at:
    grid_max_params = param_grid[np.array(real_loss).argmax(), :]
    print("Grid Max:", grid_max_params)

    return grid_max_params

def _get_SVM_params():
    C_start = 1.5
    C_end = 3
    C_step = 0.1
    C_num = ((C_end - C_start)/(C_step)) + 1
    assert np.mod(C_num, 1) == 0
    C_num = int(C_num)

    return np.linspace(C_start, C_end, C_num, True), [C_start, C_end]

def param_search_SVM():
    pcaFeatures, pcaBounds, siftFeatures, siftBounds = _get_common_params()
    lambdas, lambda_bounds = _get_SVM_params()

    param_grid = np.array([[pf, sf, C] for sf in siftFeatures for pf in pcaFeatures for C in lambdas])
    bounds = np.array([pcaBounds, siftBounds, lambda_bounds])
    
    grid_max_params = param_search(param_grid, bounds, classifier='svm')

def _get_MLP_params():
    h_start = 0
    h_end = 0
    h_step = 1
    h_num = ((h_end - h_start)/(h_step)) + 1
    assert np.mod(h_num, 1) == 0
    h_num = int(h_num)

    return np.linspace(h_start, h_end, h_num, True), [h_start, h_end]

def param_search_MLP():
    pcaFeatures, pcaBounds, siftFeatures, siftBounds = _get_common_params()
    num_hidden, hidden_bounds = _get_MLP_params()

    param_grid = np.array([[pf, sf, h] for sf in siftFeatures for pf in pcaFeatures for h in num_hidden])
    bounds = np.array([pcaBounds, siftBounds, hidden_bounds])
    
    grid_max_params = param_search(param_grid, bounds, classifier='mlp') # [10., 1550., 7.]

def _get_RF_params():
    e_start = 60
    e_end = 70
    e_step = 1
    e_num = ((e_end - e_start)/(e_step)) + 1
    assert np.mod(e_num, 1) == 0
    e_num = int(e_num)
    return np.linspace(e_start, e_end, e_num, True), [e_start, e_end]

def param_search_RF():
    pcaFeatures, pcaBounds, siftFeatures, siftBounds = _get_common_params()
    n_estimators, estimator_bounds = _get_RF_params()

    param_grid = np.array([[pf, sf, e] for sf in siftFeatures for pf in pcaFeatures for e in n_estimators])
    bounds = np.array([pcaBounds, siftBounds, estimator_bounds])
    
    grid_max_params = param_search(param_grid, bounds, classifier='rf')

def param_search_Vote():
    pcaFeatures, pcaBounds, siftFeatures, siftBounds = _get_common_params()

    lambdas, lambda_bounds = _get_SVM_params()
    num_hidden, hidden_bounds = _get_MLP_params()
    n_estimators, estimator_bounds = _get_RF_params()

    param_grid = np.array([[pf, sf, l, h, e] for sf in siftFeatures for pf in pcaFeatures for l in lambdas for h in num_hidden for e in n_estimators])
    bounds = np.array([pcaBounds, siftBounds, lambda_bounds, hidden_bounds, estimator_bounds])
    
    grid_max_params = param_search(param_grid, bounds, classifier='vote')

def eval():
    # print("Train: ", sample_loss([10., 1550., 7.], score='train', classifier='svm', eval_fn=f1_score))
    # print("Valid: ", sample_loss([10., 1550., 7.], classifier='svm', eval_fn=f1_score, show_confusion=True, random_state=12345))

    print("Train: ", sample_loss([ 20., 1350., 0.], score='train', classifier='mlp', eval_fn=f1_score))
    print("Valid: ", sample_loss([ 20., 1350., 0.], classifier='mlp', eval_fn=f1_score, show_confusion=True))

    # print("Train: ", sample_loss([ 10., 700., 77.], score='train', classifier='rf', eval_fn=f1_score))
    # print("Valid: ", sample_loss([ 10., 700., 77.], classifier='rf', eval_fn=f1_score, show_confusion=True, random_state=12345))

    # print("Train: ", sample_loss([ 10., 1150., 2.9, 0., 69.], score='train', classifier='vote', eval_fn=f1_score))
    # print("Valid: ", sample_loss([ 10., 1150., 2.9, 0., 69.], classifier='vote', eval_fn=f1_score, show_confusion=True))

    # print("Train: ", sample_loss([ 20., 1350., 0.], score='train', classifier='bagging-mlp', eval_fn=f1_score))
    # print("Valid: ", sample_loss([ 20., 1350., 0.], classifier='bagging-mlp', eval_fn=f1_score, show_confusion=True))

if __name__=="__main__":
    # param_search_SVM()
    # param_search_MLP()
    # param_search_RF()
    # param_search_Vote()
    eval()

    