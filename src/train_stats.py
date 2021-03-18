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
from processing import process_ds

from gp import bayesian_optimisation
from plotters import plot_iteration


def sample_loss(params, data=None):
    C = 2**params[0]
    nfeatures=params[1]



    # x_train, y_train, _ = process_ds(get_image_generator('train' , xray_type=XRAYTYPE.FOREARM), 10)
    # x_val, y_val, _ = process_ds(get_image_generator('valid' , xray_type=XRAYTYPE.FOREARM), 10)
    # print(x_train.shape, y_train.shape, 10)
    # print(x_val.shape, y_val.shape)
    # assert len(x_train.shape) == 2
    # assert len(y_train.shape) == 1
    # mean = np.mean(x_train, axis=0)
    # std = np.std(x_train, axis=0)
    # x_train = (x_train.astype(np.float64) - mean) / std
    # x_val = (x_val.astype(np.float64) - mean) / std

    # np.save(os.path.join(ds_file_dir, 'x_train.npy'), x_train)
    # np.save(os.path.join(ds_file_dir, 'y_train.npy'), y_train)
    # np.save(os.path.join(ds_file_dir, 'x_val.npy'), x_val)
    # np.save(os.path.join(ds_file_dir, 'y_val.npy'), y_val)
    if data is None:
        xr_type = XRAYTYPE.FOREARM
        ds_file_dir = os.path.join(os.getcwd(), 'data', 'npy_files', xr_type.value)
        os.makedirs(ds_file_dir, exist_ok=True) 
        x_train = np.load(os.path.join(ds_file_dir, 'x_train.npy'))
        y_train = np.load(os.path.join(ds_file_dir, 'y_train.npy'))
        x_val = np.load(os.path.join(ds_file_dir, 'x_val.npy'))
        y_val = np.load(os.path.join(ds_file_dir, 'y_val.npy'))
        pca = PCA(n_components=int(nfeatures))
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_val = pca.transform(x_val)
    else:
        x_train, y_train, x_val, y_val = data



    clf = svm.SVC(C=C, kernel='rbf', random_state=12345)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_train)
    # return accuracy_score(y_train, y_pred)
    return f1_score(y_train, y_pred)
    # return cohen_kappa_score(y_train, y_pred)
    # y_pred = clf.predict(x_val)
    # return accuracy_score(y_val, y_pred)
    # return f1_score(y_val, y_pred)
    # return cohen_kappa_score(y_val, y_pred)


if __name__=="__main__":
    C_start = 0
    C_end = 20
    C_num = (C_end - C_start + 1)
    nf_start = 1
    nf_end = 100
    nf_num = int((nf_end - nf_start + 1)/20.)


    lambdas = np.linspace(C_start, C_end, C_num, True)
    nfeatures = np.linspace(nf_start, nf_end, nf_num, True, dtype=np.int)

    # xr_type = XRAYTYPE.FOREARM
    # ds_file_dir = os.path.join(os.getcwd(), 'data', 'npy_files', xr_type.value)
    # os.makedirs(ds_file_dir, exist_ok=True) 
    # x_train = np.load(os.path.join(ds_file_dir, 'x_train.npy'))
    # y_train = np.load(os.path.join(ds_file_dir, 'y_train.npy'))
    # x_val = np.load(os.path.join(ds_file_dir, 'x_val.npy'))
    # y_val = np.load(os.path.join(ds_file_dir, 'y_val.npy'))
    # pca = PCA(n_components=50)  
    # pca.fit(x_train)
    # x_train = pca.transform(x_train)
    # x_val = pca.transform(x_val)
    # data = x_train, y_train, x_val, y_val
    # data = None

    # We need the cartesian combination of these two vectors
    # param_grid = np.array([[C, nf] for nf in nfeatures for C in lambdas])
    # real_loss = []
    # for params in tqdm(param_grid):
    #     real_loss.append(sample_loss(params, data))

    # # The maximum is at:
    # print(param_grid[np.array(real_loss).argmax(), :])

    # # rc('text', usetex=True)

    # C, NF = np.meshgrid(lambdas, nfeatures)
    # plt.figure()
    # cp = plt.contourf(C, NF, np.array(real_loss).reshape(C.shape))
    # plt.colorbar(cp)
    # # plt.title('Filled contours plot of loss function $\mathcal{L}$($\gamma$, $C$)')
    # plt.xlabel('$C$')
    # plt.ylabel('Number of PCA components')
    # # plt.savefig('/Users/thomashuijskens/Personal/gp-optimisation/figures/real_loss_contour.png', bbox_inches='tight')
    # plt.show()


    bounds = np.array([[C_start, C_end], [nf_start, nf_end]])

    xp, yp = bayesian_optimisation(n_iters=50, 
                               sample_loss=sample_loss, 
                               bounds=bounds,
                               n_pre_samples=3,
                               random_search=100000)
    plot_iteration(lambdas, xp, yp, 
        first_iter=3, 
        second_param_grid=nfeatures, 
        optimum=[14., 25.], 
        filepath=os.path.join(os.getcwd(), 'imgs', datetime.now().strftime('%d %m %Y %H-%M-%S'))
    )