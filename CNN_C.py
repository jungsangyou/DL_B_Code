from keras import datasets
import keras
assert keras.backend.image_data_format() == 'channels_last'

import CNN_C_Machine

class Machine(CNN_C_Machine.Machine) :
    def __init__(self):
        (X,y), (x_test, y_test) = datasets.cifar10.load_data()
        CNN_C_Machine.Machine.__init__(self, X, y, nb_classes=10)


def main() :
    m  = Machine()
    m.run()


if __name__ == '__main__' :
    main()