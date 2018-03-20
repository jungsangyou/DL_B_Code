
# _*_ coding: utf-8 _*_

import numpy as np
from keras import datasets
from keras.utils import np_utils
from keras import layers, models

class DNN(models.Sequential) :

    def __init__(self, Nin, Nh_l, Pd_l, Nout):
        models.Sequential.__init__(self)
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dropout(Pd_l[0]))
        self.add(layers.Dense(Nh_l[1], activation='relu', name='Hidden-2'))
        self.add(layers.Dropout(Pd_l[1]))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def Data_func() :

    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H, C = X_train.shape

    print('input shape ->', X_train.shape[1])

    X_train = X_train.reshape(-1, W * H * C)
    X_test = X_test.reshape(-1, W * H * C)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return (X_train, Y_train), (X_test, Y_test)

def main():

    Nh_l = [100, 50]
    Pd_l = [0.02, 0.5]
    number_of_class = 10
    Nout = number_of_class

    (X_train, Y_train), (X_test, Y_test) = Data_func()

    model = DNN(X_train.shape[1], Nh_l, Pd_l, Nout)


    history = model.fit(X_train, Y_train, epochs=10, batch_size=100, validation_split=0.2)
    performace_test = model.evaluate(X_test, Y_test, batch_size=100)
    print('Test Loss and Accuracy ->', performace_test)


if __name__ == '__main__' :
    main()