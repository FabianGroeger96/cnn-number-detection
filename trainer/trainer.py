import pickle
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, InputLayer
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical


class Trainer:

    def __init__(self):
        self.BATCH_SIZE = 16
        self.EPOCHS = 10
        self.VALIDATION_SPLIT = 0.3

        self.model = None
        self.optimizer = Adam(lr=1e-3) # specify learning rate for optimizer

        pickle_in = open("../X.pickle", "rb")
        self.X = pickle.load(pickle_in)

        pickle_in = open("../y.pickle", "rb")
        self.y = pickle.load(pickle_in)
        # one hot encode the labels
        self.y = to_categorical(self.y)

        # first we normalize the data
        self._normalize_data()

    def set_batch_size(self, batch_size):
        self.BATCH_SIZE = batch_size

    def set_epochs(self, epochs):
        self.EPOCHS = epochs

    def set_validation_split(self, validation_split):
        self.VALIDATION_SPLIT = validation_split

    def _normalize_data(self):
        self.X = self.X / 255

    def create_model(self):
        # create model
        self.model = Sequential()

        # add model layers
        # 1. Layer
        self.model.add(InputLayer(input_shape=[28, 28, 3])) # 3 because it is rgb, if gray: 1
        self.model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same'))
        self.model.add(Activation(activation='relu'))
        self.model.add(MaxPooling2D(pool_size=3, padding='same'))

        # 2. Layer
        self.model.add(Conv2D(filters=50, kernel_size=3, strides=1, padding='same'))
        self.model.add(Activation(activation='relu'))
        self.model.add(MaxPooling2D(pool_size=3, padding='same'))

        # 3. Layer
        self.model.add(Conv2D(filters=80, kernel_size=3, strides=1, padding='same'))
        self.model.add(Activation(activation='relu'))
        self.model.add(MaxPooling2D(pool_size=3, padding='same'))

        self.model.add(Dropout(rate=0.25))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(11, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer,
                      metrics=['accuracy'])

        self.model.summary()

    def create_model_light(self):
        # create model
        self.model = Sequential()

        # add model layers
        self.model.add(Conv2D(256, kernel_size=3, input_shape=(28, 28, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, kernel_size=3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())

        self.model.add(Dense(64))

        self.model.add(Dense(11))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

        self.model.summary()

    def fit_model(self):
        self.model.fit(self.X, self.y,
                       batch_size=self.BATCH_SIZE,
                       epochs=self.EPOCHS,
                       validation_split=self.VALIDATION_SPLIT)

    def save_model(self):
        model_path = "../number_detection_model.h5"
        tf.keras.models.save_model(
            self.model,
            model_path,
            overwrite=True,
            include_optimizer=True)

        print('[INFO] successfully saved model to: ', model_path)
