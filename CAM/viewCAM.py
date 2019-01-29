from keras.models import load_model
from keras.models import Model

import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import cv2
import time

if __name__ == '__main__':

    model = load_model("weed_cam_bestacc.hdf5")
    model.summary()
    
    weights = model.layers[-3].get_weights()[0]
    prepared_model = Model(model.input,(model.layers[-5].output, model.output))
    imgs = list(paths.list_images("./"))
    
    for i,fname in enumerate(imgs):
        startTime = time.time()
        image = cv2.imread(fname)
        imOrig = image.copy()
        imOrig = cv2.cvtColor(imOrig,cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        #image = cv2.resize(image,None,fx=1/4,fy=1/4)
        image = image / 255.0
        image = image[np.newaxis,:]
        feature_map, prob = prepared_model.predict(image)
        print(time.time()-startTime)
        
        bestProb = prob.argmax()
        bestProb = 0
        wProb = weights[:,bestProb]
        print(prob)
        
        for k in range(1024):
            feature_map[0,...,k] = feature_map[0,...,k]*wProb[k]
        fm = feature_map[0,:]
        fm = fm[:,:].sum(axis=2)
        fm -=fm.min()
        fm /=fm.max()
        fm *= 255.0
        fm = np.uint8(fm)
        fm = cv2.resize(fm,(imOrig.shape[1],imOrig.shape[0]))
        
        fig, (ax, ax2) = plt.subplots(1,2, figsize=(20,10))
        ax.imshow(imOrig, alpha=0.5)
        ax.imshow(fm, cmap='jet', alpha=0.5)
        ax2.imshow(imOrig)
        fig.savefig('fig{}.jpg'.format(i))