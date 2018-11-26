# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:13:49 2017

@author: Rodrigo.Andrade

ATENÇÃO: Este treinamento utiliza o generator do keras, se o treinamento for
utilizando imagens rgb a entrada da rede será RGB (obvio, não?), mas lembre que
o opencv funciona com BGR, utilize a função cv2.cvtColor para converter de RGB2BGR
"""

from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from models import Darknet19 as cnnModel
from imutils import paths
import os
import json

# configurações da rede
prefix = "weed_cam"  # nome para salvar
epochs = 10000
batchSize = 32
width = 256
height = 256
depth = 3


# path e nomes do dataset
datasetTrainPath = "/home/rodrigo/Documents/weed-detection-in-soybean-crops/train"
datasetValPath = "/home/rodrigo/Documents/weed-detection-in-soybean-crops/val"

classesTrain = next(os.walk(datasetTrainPath))[1]
classesVal = next(os.walk(datasetValPath))[1]

if not classesVal == classesTrain:
    raise Exception("As classes de treino são diferentes das classes de validação")
else:
    pastas = classesTrain


# config os geradores de dataset do keras
# caso haja apenas duas classes, a rede será uma rede de classificação binária (um unico neurônio de saída)
if len(pastas) == 2:
    classes = 1
else:
    classes = len(pastas)

# faz a leitura do nome de todos arquivos para ter a contagem de amostras
imagesTrainPaths = []
imagesValPaths = []
for pasta in pastas:
    imagesTrainPaths += list(paths.list_images(os.path.join(datasetTrainPath, pasta)))
    imagesValPaths += list(paths.list_images(os.path.join(datasetValPath, pasta)))
print(len(imagesValPaths),len(imagesTrainPaths))

trainDatagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

trainGenerator = trainDatagen.flow_from_directory(
    datasetTrainPath,
    color_mode="grayscale" if depth == 1 else "rgb",
    target_size=(height, width),
    batch_size=batchSize,
    class_mode="binary" if classes == 1 else "categorical")

valDatagen = ImageDataGenerator(rescale=1. / 255)

valGenerator = valDatagen.flow_from_directory(
    datasetValPath,
    color_mode="grayscale" if depth == 1 else "rgb",
    target_size=(height, width),
    batch_size=batchSize,
    class_mode="binary" if classes == 1 else "categorical")

with open("classIndicesTrain.txt", "w") as file:
    print("indice de classes data treino:\n", trainGenerator.class_indices)
    file.write(json.dumps(trainGenerator.class_indices))
with open("classIndicesVal.txt", "w") as file:
    print("indice de classes data validação:\n", valGenerator.class_indices)
    file.write(json.dumps(valGenerator.class_indices))

# callbacks
checkPointSaverBest = ModelCheckpoint(prefix+"_bestacc.hdf5", monitor='val_acc', verbose=1,
                                      save_best_only=True, save_weights_only=False, mode='auto', period=1)
checkPointSaver = ModelCheckpoint(prefix + "_ckp.hdf5", verbose=1,
                                  save_best_only=False, save_weights_only=False, period=10)

tb = TensorBoard(log_dir='logsTB', histogram_freq=0, batch_size=batchSize, write_graph=True,
                 write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                 embeddings_metadata=None)

if __name__ == "__main__":
    pass

# criação da rede
opt = Adadelta()
model = cnnModel.build(width=None, height=None, depth=depth, classes=classes)
model.compile(loss="binary_crossentropy" if classes == 1 else "categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])
model.summary()
model.fit_generator(
    trainGenerator,
    steps_per_epoch=len(imagesTrainPaths) // batchSize,
    epochs=epochs,
    validation_data=valGenerator,
    validation_steps=len(imagesValPaths) // batchSize,
    callbacks=[checkPointSaverBest, checkPointSaver, tb],
    workers=8,
    max_queue_size=40)







