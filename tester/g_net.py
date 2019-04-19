import constants
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adam


def load_model(weights_path=None):
    # create model
    model = Sequential()

    # add model layers
    # 1. Layer
    model.add(Conv2D(filters=16, kernel_size=3, strides=1, padding='same',
                          input_shape=(constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION)))
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=3, padding='same'))

    # 2. Layer
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same'))
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=3, padding='same'))

    # 3. Layer
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same'))
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=3, padding='same'))

    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(rate=0.5))
    model.add(Dense(len(constants.CATEGORIES), activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    return model
