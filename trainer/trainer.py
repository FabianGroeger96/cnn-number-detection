import os
import pickle
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D


class Trainer:

    def __init__(self):
        self.BATCH_SIZE = 16
        self.EPOCHS = 10
        self.VALIDATION_SPLIT = 0.3

        self.model = None

        pickle_in = open("../X.pickle", "rb")
        self.X = pickle.load(pickle_in)

        pickle_in = open("../y.pickle", "rb")
        self.y = pickle.load(pickle_in)

    def set_batch_size(self, batch_size):
        self.BATCH_SIZE = batch_size

    def set_epochs(self, epochs):
        self.EPOCHS = epochs

    def set_validation_split(self, validation_split):
        self.VALIDATION_SPLIT = validation_split

    def _normalize_data(self):
        self.X = self.X / 255

    def create_model(self):
        # first we normalize the data
        self._normalize_data()

        # create model
        self.model = Sequential()

        # add model layers
        self.model.add(Conv2D(256, (3, 3), input_shape=(25, 25, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        #self.model.add(Dense(64))

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

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

        print('successfully saved model to: ', model_path)
