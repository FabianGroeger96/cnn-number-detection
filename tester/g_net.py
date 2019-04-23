import constants
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adam


def load_model(weights_path=None):
    # create model
    model = Sequential()

    # add model layers
    model.add(Conv2D(16, kernel_size=3, input_shape=(constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(rate=0.5))
    model.add(Flatten())

    model.add(Dense(len(constants.CATEGORIES)))
    model.add(Activation('softmax'))

    if weights_path:
        model.load_weights(weights_path)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    return model
