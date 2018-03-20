
# _*_ coding: utf-8 _*_

Nin = 784
Nh_l = [100, 50]
number_of_class = 10
Nout = number_of_class

from keras import layers, models

class DNN(models.Sequential) :
    def __init__(self, Nin, Nh_l, Nout):
        models.Sequential.__init__(self)
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dense(Nh_l[1], activation='relu', name='Hidden-2'))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def Data_func() :
    import numpy as np
    from keras import datasets
    from keras.utils import np_utils

    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X_test = X_test.reshape(-1, W * H)

    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return (X_train, Y_train), (X_test, Y_test)

def main():
    model = DNN(Nin, Nh_l, Nout)
    (X_train, Y_train), (X_test, Y_test) = Data_func()

    history = model.fit(X_train, Y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=2)
    performace_test = model.evaluate(X_test, Y_test, batch_size=100)
    print('Test Loss and Accuracy ->', performace_test)


if __name__ == '__main__' :
    main()