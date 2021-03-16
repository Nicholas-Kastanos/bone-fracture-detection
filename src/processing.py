import sys

import cv2 as cv
import numpy as np
from scipy.ndimage import grey_closing, grey_opening

from dataset import XRAYTYPE, get_image_generator


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
        desc = desc[:100, :]
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


def extract_features(img: np.ndarray, show_imgs:bool = False):
    img = 255-img
    if show_imgs:
        show("IMG", img)

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

    ret, kp, desc = sift_features(img)
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

def process_ds(generator, max_length=None):
    features = []
    for imgs, label in generator:
        features_arr = []
        for img in imgs:
            ret, f = extract_features(img)
            if ret:
                features_arr.append(f)
        if len(features_arr)>0:
            features.append((features_arr, label))
    if max_length is None:
        max_length = max_len(max(features, key=max_len))
    full_features = []
    labels = []
    for f, l in features:
        ff = fill_features(f, max_length)
        full_features.append(ff)
        labels.append(l)
    full_features = np.asarray(full_features)
    labels = np.array(labels)
    return full_features, labels, max_length

if __name__ == "__main__":
    gen = get_image_generator('train' , xray_type=XRAYTYPE.FOREARM)
    x, y, max_length = process_ds(gen)
    print(x.shape, y.shape, max_length)
