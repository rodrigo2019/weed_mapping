# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:47:15 2019

@author: ANO8CA
"""
from imutils import paths
from tqdm import tqdm
import numpy as np
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to the image or folder")
args = vars(ap.parse_args())

if __name__ == "__main__":
    slic_types = {0: cv2.ximgproc.SLIC, 1: cv2.ximgproc.SLICO, 2: cv2.ximgproc.MSLIC}
    while True:
        algorithm_type = input("Which algorithm?\n"
                              "[0] SLIC\n"
                              "[1] SLICO\n"
                              "[2] MSLIC (broken)\n"
                              "[3] LSC\n")
        algorithm_type = int(algorithm_type)
        if algorithm_type < 0 or algorithm_type > 3:
            print("Choose between 0 and 3")
        else:
            break

    if os.path.isdir(args["input"]):
        images_path = list(paths.list_images(args["input"]))
    else:
        images_path = args["input"]
    if not os.path.isdir("./segmented"):
        os.mkdir("./segmented")

    for fname in tqdm(images_path):
        basename = os.path.basename(fname)
        basename = os.path.splitext(basename)[0]

        if not os.path.isdir("./segmented/{}".format(basename)):
            os.mkdir("./segmented/{}".format(basename))
        else:
            print("folder already exists!\nIgnoring this file")
            continue

        image = cv2.imread(fname)
        blured = cv2.GaussianBlur(image, (3, 3), 0)
        cieLab = cv2.cvtColor(blured, cv2.COLOR_BGR2Lab)
        if algorithm_type <= 2:
            super_pixel = cv2.ximgproc.createSuperpixelSLIC(cieLab, algorithm=slic_types[algorithm_type], region_size=50,
                                                            ruler=10.0)
        elif algorithm_type == 3:
            super_pixel = cv2.ximgproc.createSuperpixelLSC(cieLab, region_size=50, ratio=0.075)

        super_pixels_qty = super_pixel.getNumberOfSuperpixels()
        super_pixel.iterate(num_iterations=10)
        labels = super_pixel.getLabels()
        for i in tqdm(range(super_pixels_qty)):
            mask = labels != i
            mask = mask[..., np.newaxis]
            mask = np.tile(mask, (1, 1, 3))
            image_crop = image.copy()
            image_crop[mask] = 0
            y, x = np.where(labels == i)
            xmin = min(x)
            xmax = max(x)
            ymin = min(y)
            ymax = max(y)
            image_crop = image_crop[ymin:ymax, xmin: xmax]
            cv2.imwrite("./segmented/{}/{}.jpg".format(basename, i), image_crop)
