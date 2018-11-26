from imutils import paths
from tqdm import tqdm
import os
import cv2


imgs_path = list(paths.list_images("/home/rodrigo/Documents/weed-detection-in-soybean-crops/"))

for fname in tqdm(imgs_path):
    image = cv2.imread(fname)
    os.remove(fname)
    cv2.imwrite(fname[:-4]+".jpg",image)