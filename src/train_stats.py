import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             plot_confusion_matrix, f1_score)
from tqdm import tqdm

from dataset import XRAYTYPE, get_image_generator
from processing import process_ds

if __name__=="__main__":

    xr_type = XRAYTYPE.FOREARM
    ds_file_dir = os.path.join(os.getcwd(), 'data', 'npy_files', xr_type.value)
    os.makedirs(ds_file_dir, exist_ok=True) 

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

    x_train = np.load(os.path.join(ds_file_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(ds_file_dir, 'y_train.npy'))
    x_val = np.load(os.path.join(ds_file_dir, 'x_val.npy'))
    y_val = np.load(os.path.join(ds_file_dir, 'y_val.npy'))

    pca = PCA(n_components=50)
    pca.fit(x_train)
    # plt.plot(pca.explained_variance_ratio_)
    # plt.show()
    x_train = pca.transform(x_train)
    x_val = pca.transform(x_val)

    start = -5
    end = 15
    num = (end - start + 1) * 10

    a_train = []
    a_val = []
    f1_train = []
    f1_val = []

    ls = np.logspace(start, end, num, True, 2.)

    for C in tqdm(ls):
        clf = svm.SVC(C=C, kernel='rbf')
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_train)
        a_train.append(accuracy_score(y_train, y_pred))
        f1_train.append(f1_score(y_train, y_pred))
        # print("Train Accuracy C="+str(C), a_train[-1])
        y_pred = clf.predict(x_val)
        a_val.append(accuracy_score(y_val, y_pred))
        f1_val.append(f1_score(y_val, y_pred))
        # print("Validation Accuracy C="+str(C), a_val[-1])
        # print(confusion_matrix(y_val, y_pred))
    # plot_confusion_matrix(clf, x_val, y_val)  

    idx = np.argmax(f1_val)
    print("Best: ", "C", ls[int(idx)], "f1:", f1_train[int(idx)], "val_f1:", f1_val[int(idx)])
    plt.plot(a_train)
    plt.plot(a_val)
    plt.plot(f1_train)
    plt.plot(f1_val)
    plt.legend(['acc', 'val_acc', 'f1', 'val_f1'])
    plt.show()  
