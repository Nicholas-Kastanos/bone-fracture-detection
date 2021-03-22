import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, plot_confusion_matrix)
from tqdm import tqdm
from datetime import datetime

from dataset import XRAYTYPE, get_image_generator
from processing import load_ds

from gp import bayesian_optimisation
from plotters import plot_iteration


def sample_loss(params, data=None):
    C = 2**params[0]
    n_pca_components=int(params[1])
    n_SIFT_features=int(params[2])

    if data is None:
        xr_type = XRAYTYPE.FOREARM
        x_train, y_train, x_val, y_val = load_ds(xr_type, n_pca=n_pca_components, n_SIFT_features=n_SIFT_features)
    else:
        x_train, y_train, x_val, y_val = data



    clf = svm.SVC(C=C, kernel='rbf', random_state=12345)
    clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_train)
    # return accuracy_score(y_train, y_pred)
    # return f1_score(y_train, y_pred)
    # return cohen_kappa_score(y_train, y_pred)
    y_pred = clf.predict(x_val)
    # return accuracy_score(y_val, y_pred)
    return f1_score(y_val, y_pred)
    # return cohen_kappa_score(y_val, y_pred)


if __name__=="__main__":

    start_time = datetime.now()

    C_start = 0
    C_end = 20
    C_num = int((C_end - C_start + 1)/2.)
    p_start = 1
    p_end = 100
    p_num = int((p_end - p_start + 1)/10.)
    s_start = 50
    s_end= 2000
    s_num=int((s_end - s_start + 1)/50.)


    lambdas = np.linspace(C_start, C_end, C_num, True)
    pcaFeatures = np.linspace(p_start, p_end, p_num, True, dtype=np.int)
    siftFeatures = np.linspace(s_start, s_end, s_num, True, dtype=np.int)


    # We need the cartesian combination of these two vectors
    param_grid = np.array([[C, pf, sf] for sf in siftFeatures for pf in pcaFeatures for C in lambdas])
    real_loss = []
    print("Running Parameter Grid")
    for params in tqdm(param_grid):
        real_loss.append(sample_loss(params))

    # The maximum is at:
    max_params = param_grid[np.array(real_loss).argmax(), :]
    print(max_params) # >> [   6.66666667   12.         1178.        ]

    C, PF, SF = np.meshgrid(lambdas, pcaFeatures, siftFeatures)
    plt.figure()
    # cp = plt.contourf(C, PF, np.array(real_loss).reshape(C.shape))
    cp = plt.contour3D(C, PF, SF, np.array(real_loss).reshape(C.shape))
    plt.colorbar(cp)
    # plt.title('Filled contours plot of loss function $\mathcal{L}$($\gamma$, $C$)')
    # plt.xlabel('$C$')
    # plt.ylabel('Number of PCA components')
    plt.scatter(max_params[0], max_params[1], max_params[2], marker='*', c='red', s=150)
    fig_dir = os.path.join(os.getcwd(), 'imgs', start_time.strftime('%d %m %Y %H-%M-%S')+ ' f1_val')
    grid_fig_dir = os.path.join(fig_dir, 'grid')
    os.makedirs(grid_fig_dir, exist_ok=True)
    plt.savefig(os.path.join(grid_fig_dir, 'final.png'), bbox_inches='tight')

    # print("Running BO")
    # bounds = np.array([[C_start, C_end], [p_start, p_end], [s_start, s_end]])
    # x0 = np.array([[C, nf] for nf in nfeatures[::2] for C in lambdas[::2]])
    # print(len(x0))
    # xp, yp = bayesian_optimisation(n_iters=100, 
    #                            sample_loss=sample_loss, 
    #                            bounds=bounds,
    #                            x0=x0,
    #                            n_pre_samples=len(x0),
    #                         #    random_search=None)
    #                            random_search=100000)
    # plot_iteration(lambdas, xp, yp, 
    #     first_iter=len(x0), 
    #     second_param_grid=nfeatures, 
    #     optimum=max_params, 
    #     filepath=os.path.join(fig_dir, 'bo')
    # )