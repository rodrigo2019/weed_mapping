from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

class Alexnet:

    def build(width = 224, height = 224, depth = 3, classes=10, l2_reg=0., weights=None):
        img_shape = (height, width, depth)
        # Initialize model
        alexnet = Sequential()

        # Layer 1
        alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape, padding='same', kernel_regularizer=l2(l2_reg)))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 2
        alexnet.add(Conv2D(256, (5, 5), padding='same'))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 3
        alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(512, (3, 3), padding='same'))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 4
        alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(1024, (3, 3), padding='same'))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))

        # Layer 5
        alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(1024, (3, 3), padding='same'))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 6
        alexnet.add(Flatten())
        alexnet.add(Dense(3072))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(Dropout(0.5))

        # Layer 7
        alexnet.add(Dense(4096))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(Dropout(0.5))

        # Layer 8
        alexnet.add(Dense(classes))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('softmax'))

        if weights is not None:
            alexnet.load_weights(weights)

        return alexnet


class Darknet19:

    def build(width=224, height=224, depth=3, classes=1000, weightsPath=None):

        input_shape = (height, width, depth)

        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(64, (1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(128, (1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(256, (1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(256, (1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(1024, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, (1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(1024, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, (1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(1024, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        '''model.add(Conv2D(classes, (1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))'''

        model.add(GlobalAveragePooling2D(data_format="channels_last"))

        model.add(Dense(classes))
        model.add(BatchNormalization())
        model.add(Activation("sigmoid"))

        # model.add(Dropout(0.5)) #Não há dropout no paper original
        # model.add(Dense(classes))
        '''if classes == 1:
            model.add(Activation("sigmoid"))
        else:
            model.add(Activation("softmax"))'''

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
