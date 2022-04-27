import cv2
import numpy as np


def blur(img):
    output = cv2.GaussianBlur(img, (7, 7), cv2.BORDER_DEFAULT)
    return output


def dilated(img):
    output = cv2.dilate(img, (7, 7), iterations=2)
    return output


def canny(img):
    output = cv2.Canny(img, 125, 175)
    return output


def thresh(img):
    threshold, output = cv2.threshold(img, 100, 150, cv2.THRESH_BINARY)
    return output


def bilateral(img):
    output = cv2.bilateralFilter(img, d=9, sigmaColor=200, sigmaSpace=200)
    return output

""" error
def watercolor(img):
    gray_1 = cv2.medianBlur(img, 5)
    edges = cv2.adaptiveThreshold(gray_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    return edges


def cartoon(img):
    gray_1 = cv2.medianBlur(img, 5)
    edges = cv2.adaptiveThreshold(gray_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    output = cv2.bitwise_and(bilateral, bilateral, mask=edges)
    return output
"""

def invert(img):
    output = cv2.bitwise_not(img)
    return output


def smoothing(img):
    invert1 = cv2.bitwise_not(img)
    output = cv2.GaussianBlur(invert1, (21, 21), sigmaX=0, sigmaY=0)
    return output


def pencil_sketch_grey(img):
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return sk_gray


def pencil_sketch_col(img):
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return sk_color


def glow(img):
    glow_strength = 1
    glow_radius = 25
    img_blurred = cv2.GaussianBlur(img, (glow_radius, glow_radius), 1)
    output = cv2.addWeighted(img, 1, img_blurred, glow_strength, 0)
    return output


def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
    output = cv2.filter2D(img, -1, kernel)
    return output

def binary(img):
    ret, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return bw
