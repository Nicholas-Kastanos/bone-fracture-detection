import sys
import os
import gc

import cv2 as cv
import numpy as np
from scipy.ndimage import grey_closing, grey_opening
from sklearn.decomposition import PCA

import utils
from dataset import XRAYTYPE, get_image_generator, get_max_images_per_study


def blur(img):
    return cv.GaussianBlur(img, (3, 3), 0)

def grey_open_close(img):
    return grey_closing(grey_opening(img, size=(3, 3)), size=(3, 3))

def bin_open_close(img):
    return cv.morphologyEx(cv.morphologyEx(img, cv.MORPH_OPEN, (3, 3)), cv.MORPH_CLOSE, (3, 3))

def img_to_edges(img):
    return cv.Canny(img, 150, 200, apertureSize=5, L2gradient=True)

def sift_features(img, nfeatures=100):
    sift = cv.SIFT_create(nfeatures)
    kp, desc = sift.detectAndCompute(img,None)
    if len(kp) == 0:
        return False, np.empty(1), np.empty(1)
    if len(kp) > nfeatures: # OpenCV randomly gives nfeatures + 1 features
        while len(kp) > nfeatures:
            kp.pop()
        desc = desc[:nfeatures, :]
    elif len(kp) < nfeatures:
        if desc is None:
            cv.imshow("NONE", img)
            cv.waitKey(0)
        z = np.zeros((nfeatures-len(kp), desc.shape[1]))
        desc = np.concatenate([desc, z])
    assert desc.shape[0] == nfeatures
    return True, kp, desc

def show(name, img):
    sc_fac = 1.5
    cv.imshow(name, cv.resize(img, None, fx=sc_fac, fy=sc_fac))

def remove_background(img, show_imgs):
    ret, thresh = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    if show_imgs:
        show("Thresh", thresh)
    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    col_img = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    # cv.drawContours(col_img, contours, -1, (0,255,0), 3)
    largest_c = max(contours, key=cv.contourArea)
    # draw white contour on black background as mask
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv.drawContours(mask, [largest_c], 0, (255), cv.FILLED)
    # invert mask so shapes are white on black background
    mask_inv = 255 - mask
    if show_imgs:
        show("MASK", mask)
    # create new (white) background
    bckgnd = np.full_like(img, (255))
    # apply mask to image
    image_masked = cv.bitwise_and(img, img, mask=mask)
    # apply inverse mask to background
    bckgnd_masked = cv.bitwise_and(bckgnd, bckgnd, mask=mask_inv)
    # add together
    img = cv.add(image_masked, bckgnd_masked)
    return img


def extract_features(img: np.ndarray, show_imgs:bool = False, n_SIFT_features=100):
    if show_imgs:
        show("IMG", img)
    img = 255-img

    img = blur(img)
    if show_imgs:
        show("BLUR",img)

    img = grey_open_close(img)
    if show_imgs:
        show("GMORPH", img)

    img = remove_background(img, show_imgs)
    if show_imgs:
        show("FORGROUND", img)

    img = img_to_edges(img)
    if show_imgs:
        show("CANNY", img)

    ret, kp, desc = sift_features(img, n_SIFT_features)
    if not ret:
        return False, np.empty(1)
    if show_imgs:
        sift_img = img.copy()
        sift_img = cv.drawKeypoints(img, kp, sift_img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        show("SIFT", sift_img)
    
    
    if show_imgs:
        cv.waitKey(0)
    return True, np.ravel(desc)

def fill_features(features_arr, length: int):
    assert length > 0
    feature_vec_len = features_arr[0].shape[0]
    assert feature_vec_len > 0
    np_features = np.concatenate([np.array(v).reshape((1, -1)) for v in features_arr], axis=0)
    if np_features.shape[0] < length:
        z = np.zeros((length - np_features.shape[0], np_features.shape[1]))
        np_features = np.concatenate([np_features, z])
    rng = np.random.default_rng()
    rng.shuffle(np_features, axis=0)
    return np.ravel(np_features)

def max_len(x):
    return len(x[0])

def _calc_features(generator, n_SIFT_features=100, show_imgs=False):
    features = []
    for imgs, label in generator:
        features_arr = []
        for img in imgs:
            ret, f = extract_features(img, n_SIFT_features=n_SIFT_features, show_imgs=show_imgs)
            if ret:
                features_arr.append(f)
        if len(features_arr)>0:
            features.append((features_arr, label))
    return features

def _apply_or_fit_PCA(features, pca):
    # Combine into a single array, but remember which study the images belonged to.
    num_images = []
    labels = []
    image_features = []
    for f, l in features:
        labels.append(l)
        num_images.append(len(f))
        image_features = image_features + f

    # Apply or fit PCA
    if not isinstance(pca, PCA):
        pca = PCA(n_components=pca)
        pca.fit(image_features)
    image_features = pca.transform(image_features)

    # Undo concatenation
    features = []
    for num in num_images:
        study_features = []
        for _ in range(num):
            study_features.append(image_features[0])
            image_features = image_features[1:]
        features.append((study_features, labels[0]))
        labels.pop(0)
    return features, pca

def _concat_feature_vectors(features, max_length=None):
    if max_length is None:
        max_length = max_len(max(features, key=max_len))
    full_features = []
    labels = []
    for f, l in features:
        ff = fill_features(f, max_length)
        full_features.append(ff)
        labels.append(l)
    return np.asarray(full_features), np.array(labels)

def _files_exist(files):
    return all(os.path.isfile(f) for f in files)

def load_ds(xray_type, n_pca=None, n_SIFT_features=100):
    # Check if ds with correct type, PCA amount, and features exists
    train_ds_X_path = os.path.join(os.getcwd(), 'data', 'npy_files', str(xray_type.value), 'train_ds_X_pca-'+str(n_pca)+'_sift-'+str(n_SIFT_features)+'.pkl')
    train_ds_Y_path = os.path.join(os.getcwd(), 'data', 'npy_files', str(xray_type.value), 'train_ds_Y_pca-'+str(n_pca)+'_sift-'+str(n_SIFT_features)+'.pkl')
    valid_ds_X_path = os.path.join(os.getcwd(), 'data', 'npy_files', str(xray_type.value), 'valid_ds_X_pca-'+str(n_pca)+'_sift-'+str(n_SIFT_features)+'.pkl')
    valid_ds_Y_path = os.path.join(os.getcwd(), 'data', 'npy_files', str(xray_type.value), 'valid_ds_Y_pca-'+str(n_pca)+'_sift-'+str(n_SIFT_features)+'.pkl')
    if _files_exist([train_ds_X_path, train_ds_Y_path, valid_ds_X_path, valid_ds_Y_path]):
    # If both files exist, use file contents.
        return utils.read_pickle(train_ds_X_path), \
            utils.read_pickle(train_ds_Y_path), \
            utils.read_pickle(valid_ds_X_path), \
            utils.read_pickle(valid_ds_Y_path)
    else:
    # If either file does not exist, recreate all.
        ## Check if ds with correct type and features exists
        train_features_path = os.path.join(os.getcwd(), 'data', 'npy_files', str(xray_type.value), 'train_features_sift-'+str(n_SIFT_features)+'.pkl')
        valid_features_path = os.path.join(os.getcwd(), 'data', 'npy_files', str(xray_type.value), 'valid_features_sift-'+str(n_SIFT_features)+'.pkl')
        if _files_exist([train_features_path, valid_features_path]):
        # If both files exist, use files to make ds files.
            # print("Creating ds files with n_pca="+str(n_pca)+" n_SIFT_features="+str(n_SIFT_features))
            max_features_path = os.path.join(os.getcwd(), 'data', 'npy_files', str(xray_type.value), 'max_images.pkl')
            if _files_exist([max_features_path]):
                max_images = utils.read_pickle(max_features_path)
            else:
                max_images = get_max_images_per_study(xray_type)
                utils.write_pickle(max_images, max_features_path)

            train_features = utils.read_pickle(train_features_path)
            pca=None
            if n_pca is not None: # We want to do PCA on a per image basis
                train_features, pca = _apply_or_fit_PCA(train_features, n_pca)
            train_ds_X, train_ds_Y = _concat_feature_vectors(train_features, max_images)
            utils.write_pickle(train_ds_X, train_ds_X_path)
            utils.write_pickle(train_ds_Y, train_ds_Y_path)
            del train_ds_X
            del train_ds_Y
            gc.collect()

            valid_features = utils.read_pickle(valid_features_path)
            if n_pca is not None: # We want to do PCA on a per image basis (If pca happened for train_ds, pca will have the PCA object)
                valid_features, _ = _apply_or_fit_PCA(valid_features, pca)
            valid_ds_X, valid_ds_Y = _concat_feature_vectors(valid_features, max_images)
            utils.write_pickle(valid_ds_X, valid_ds_X_path)
            utils.write_pickle(valid_ds_Y, valid_ds_Y_path)
            del valid_ds_X
            del valid_ds_Y
            gc.collect()
            return load_ds(xray_type, n_pca, n_SIFT_features)
        else:
        # If either file does not exist, recreate all.
            ### Check if ds with correct type exists
            train_images_path = os.path.join(os.getcwd(), 'data', 'npy_files', str(xray_type.value), 'train_images.pkl')
            valid_images_path = os.path.join(os.getcwd(), 'data', 'npy_files', str(xray_type.value), 'valid_images.pkl')
            if _files_exist([train_images_path, valid_images_path]):
            # If both files exist, use files to create feature files.
                # print("Creating feature files with n_SIFT_features="+str(n_SIFT_features))
                train_images = utils.read_pickle(train_images_path)
                train_features = _calc_features(train_images, n_SIFT_features)
                utils.write_pickle(train_features, train_features_path)
                del train_images
                del train_features
                gc.collect()
                valid_images = utils.read_pickle(valid_images_path)
                valid_features = _calc_features(valid_images, n_SIFT_features)
                utils.write_pickle(valid_features, valid_features_path)
                del valid_images
                del valid_features,
                gc.collect()
                return load_ds(xray_type, n_pca, n_SIFT_features) 
            else:
            # If either file does not exist, recreate it.
                # print("Creating _images.pkl")
                train_images = list(get_image_generator('train', xray_type))
                utils.write_pickle(train_images, train_images_path)
                del train_images
                gc.collect()
                valid_images = list(get_image_generator('valid', xray_type))
                utils.write_pickle(valid_images, valid_images_path)
                del valid_images
                gc.collect()
                return load_ds(xray_type, n_pca, n_SIFT_features)

if __name__ == "__main__":
    gen = get_image_generator('valid', XRAYTYPE.FOREARM)
    _calc_features(gen, show_imgs=True, n_SIFT_features=1400)
    # train_X, train_Y, valid_X, valid_Y = load_ds(XRAYTYPE.FOREARM, 50, 100)
