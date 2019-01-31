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

NUM_SUPERPIXELS = 500
NUM_ITERATIONS = 10
NUM_LEVELS = 4
PRIOR = 2
NUM_HISTOGRAM_BINS = 5

if __name__ == "__main__":
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
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width, channels = image.shape

        seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, NUM_SUPERPIXELS, NUM_LEVELS, PRIOR,
                                                   NUM_HISTOGRAM_BINS)
        seeds.iterate(hsv_image, NUM_ITERATIONS)
        super_pixels_qty = seeds.getNumberOfSuperpixels()

        labels = seeds.getLabels()
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

        # stitch foreground & background together
        mask = seeds.getLabelContourMask(False)
        color_img = np.zeros((height, width, 3), np.uint8)
        color_img[:] = (0, 0, 255)
        mask_inv = cv2.bitwise_not(mask)
        result_bg = cv2.bitwise_and(image, image, mask=mask_inv)
        result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
        result = cv2.add(result_bg, result_fg)
        cv2.imwrite("./segmented/{}/{}_{}_debug.jpg".format(basename, NUM_SUPERPIXELS, basename), result)
