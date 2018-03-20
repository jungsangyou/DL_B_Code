
# _*_ coding: utf-8 _*_
from keras import layers, models

class ANN(models.Model) :
    def __init__(self, Nin, Nh, Nout):
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')


        x = layers.Input(shape=(Nin,))
        h = relu(hidden(x))
        y = output(h)

        models.Model.__init__(self, x, y)
        self.compile(loss='mse', optimizer='sgd')


from keras import datasets
from sklearn import preprocessing

def Data_func() :
    (X_train, y_train), (X_test, y_test) = datasets.boston_housing.load_data()
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (X_train, y_train), (X_test, y_test)


def main():
    Nin = 13
    Nh = 8
    number_of_class = 1
    Nout = number_of_class

    model = ANN(Nin, Nh, Nout)
    (X_train, Y_train), (X_test, Y_test) = Data_func()

    history = model.fit(X_train, Y_train, epochs=100, batch_size=128, validation_split=0.2)
    performace_test = model.evaluate(X_test, Y_test, batch_size=128)
    print('\nTest Loss -> {:.2f}', format(performace_test))


if __name__ == '__main__' :
    main()