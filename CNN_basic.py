
# _*_ coding: utf-8 _*_

import numpy as np
import keras
from keras import datasets
from keras.utils import np_utils
from keras import layers, models, backend

class CNN(models.Sequential) :
    def __init__(self, input_shape, num_classes):
        models.Sequential.__init__(self)


        self.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
        self.add(layers.Conv2D(64, (3,3), activation='relu'))
        self.add(layers.MaxPooling2D(pool_size=(2,2)))
        self.add(layers.Dropout(0.25))
        self.add(layers.Flatten())
        self.add(layers.Dense(128, activation='relu'))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(num_classes, activation='softmax'))

        self.compile(loss=keras.losses.categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])


class DATA() :
    def __init__(self):
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        img_rows, img_cols = x_train.shape[1:]

        if backend.image_data_format() == 'channels_first' :
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

            input_shape = (1, img_rows, img_cols)

        else :
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test





def main():

    batch_size = 128
    epochs = 10

    data = DATA()

    model = CNN(data.input_shape, data.num_classes)

    history = model.fit(data.x_train, data.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, vervose=1)
    score = model.evaluate(data.x_test, data.y_test)
    print('Test Loss  ->', score[0])
    print('Test accuracy ->', score[1])


if __name__ == '__main__' :
    main()