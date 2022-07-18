import numpy as np
import cv2 as cv


def get_features(img):
    img = np.resize(img, (28, 28)).astype(np.uint8).T
    corners = cv.goodFeaturesToTrack(img, 25, 0.05, 4, useHarrisDetector=True, k=-0.09)

    try:
        corners = np.int0(corners)
    except TypeError:
        corners = np.array([])
        corn_xm, corn_ym = 0, 0
        corn_xs, corn_ys = 0, 0
    else:
        corners.resize((corners.shape[0], 2))
        corn_xm, corn_ym = corners.mean(axis=0)
        corn_xs, corn_ys = corners.std(axis=0)
    num_corners = len(corners)

    circles = cv.HoughCircles(
        img, cv.HOUGH_GRADIENT, 1, 5, param1=20, param2=8, minRadius=5, maxRadius=20
    )
    try:
        num_circles = len(circles[0, :])
    except TypeError:
        num_circles = 0
        circ_m = 0
        circ_s = 0
    else:
        circ_m = np.array(circles[0][:, 0:1]).mean()
        circ_s = np.array(circles[0][:, 2]).mean()

    edges = cv.Canny(img, 20, 180, apertureSize=3)
    lines = cv.HoughLines(edges, 0.5, np.pi / 90, 10)

    try:
        num_lines = len(lines)
    except TypeError:
        num_lines = 0
        rho_m, rho_s = 0, 0
        theta_m, theta_s = 0, 0
    else:
        lines.resize((lines.shape[0], 2))
        rho, theta = lines.T
        rho_m = np.array(rho).mean()
        theta_m = np.array(theta).mean()
        rho_s = np.array(rho).std()
        theta_s = np.array(theta).std()

    return np.array(
        [
            num_corners,
            corn_xm,
            corn_ym,
            corn_xs,
            corn_ys,
            num_circles,
            circ_m,
            circ_s,
            num_lines,
            rho_m,
            rho_s,
            theta_m,
            theta_s,
        ]  # len() = 13
    )


def get_plain_data(img):
    return img.flatten().astype(np.uint8)


def get_PCA(img):
    raise NotImplementedError
