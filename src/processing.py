from dataset import get_image_generator, XRAYTYPE
import numpy as np
import cv2 as cv
from scipy.ndimage import grey_opening, grey_closing

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
    return sift.detectAndCompute(img, None)

def show(name, img):
    sc_fac = 1.5
    cv.imshow(name, cv.resize(img, None, fx=sc_fac, fy=sc_fac))

def extract_features(img):
    img = 255-img
    show("IMG", img)
    img = blur(img)
    show("BLUR",img)
    img = grey_open_close(img)
    show("GMORPH", img)


    ret, thresh = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
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
    show("MASK", mask)
    # create new (white) background
    bckgnd = np.full_like(img, (255))
    # apply mask to image
    image_masked = cv.bitwise_and(img, img, mask=mask)
    # apply inverse mask to background
    bckgnd_masked = cv.bitwise_and(bckgnd, bckgnd, mask=mask_inv)
    # add together
    img = cv.add(image_masked, bckgnd_masked)
    show("FORGROUND", img)


    img = img_to_edges(img)
    show("CANNY", img)

    # img = bin_open_close(img)
    # show("BMORPH", img)

    kp, desc = sift_features(img)
    sift_img = img.copy()
    sift_img = cv.drawKeypoints(img, kp, sift_img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show("SIFT", sift_img)

    cv.waitKey(0)

if __name__ == "__main__":
    gen = get_image_generator(xray_type=XRAYTYPE.FOREARM)
    for _ in range(20):
        study = next(gen)
        extract_features(study[0][0])