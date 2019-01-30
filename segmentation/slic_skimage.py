# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:29:56 2019

@author: ANO8CA
"""

# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to the image or folder")
args = vars(ap.parse_args())

if __name__ == "__main__":
    if os.path.isdir(args["input"]):
        images_path = list(paths.list_images(args["input"]))
    else:
        images_path = args["input"]
    if not os.path.isdir("./segmented"):
        os.mkdir("./segmented")

    for fname in tqdm(images_path):
        # load the image and convert it to a floating point data type
        image = cv2.imread(fname)
        basename = os.path.basename(fname)
        basename = os.path.splitext(basename)[0]
        
        if not os.path.isdir("./segmented/{}".format(basename)):
            os.mkdir("./segmented/{}".format(basename))
        else:
            print("folder already exists!\nIgnoring this file")
            continue
        
        # loop over the number of segments
        for numSegments in tqdm((100, 200, 300, 400, 500, 1000, 2000)):
            # apply SLIC and extract (approximately) the supplied number
            # of segments
            segments = slic(img_as_float(image), n_segments=numSegments, sigma=5)
        
            # show the output of SLIC
            fig = plt.figure("Superpixels -- {} segments".format(numSegments))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mark_boundaries(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), segments))
            plt.axis("off")
            plt.savefig("./segmented/{}/{}_{}_debug.jpg".format(basename, numSegments, basename), dpi=1000)
        
            # loop over the unique segment values
            for (i, segVal) in enumerate(tqdm(np.unique(segments))):
                # construct a mask for the segment
                mask = np.zeros(image.shape[:2], dtype="uint8")
                mask[segments == segVal] = 255

                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(contours[0])
                mask = mask[y:y+h, x:x+w]
                image_crop = image[y:y+h, x:x+w]
             
                # show the masked region
                cv2.imwrite("./segmented/{}/{}_{}.jpg".format(basename, numSegments, i),
                            cv2.bitwise_and(image_crop, image_crop, mask=mask))
