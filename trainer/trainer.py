import pickle
import time
import constants
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard


class Trainer:

    def __init__(self):
        self.model = None
        # specify learning rate for optimizer
        self.optimizer = Adam(lr=1e-3)
        # to start tensorboard run: tensorboard --logdir=logs/, in working directory
        self.tensorboard = None

        pickle_in = open("../X.pickle", "rb")
        X = pickle.load(pickle_in)
        # scale the raw pixel intensities to the range [0, 1]
        X = X / 255

        pickle_in = open("../y.pickle", "rb")
        y = pickle.load(pickle_in)

        # train test split
        (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(X, y,
                                                          test_size=constants.VALIDATION_SPLIT,
                                                          random_state=42)
        self.lb = LabelBinarizer()
        self.trainY = self.lb.fit_transform(self.trainY)
        self.testY = self.lb.transform(self.testY)

    def create_model_deep(self):
        # give the model a name for tensorboard
        NAME = 'CNN-number-detection-deepnet'
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

        print('[INFO] creating model: ', NAME)

        # create model
        self.model = Sequential()

        # add model layers
        # 1. Layer
        self.model.add(Conv2D(filters=16, kernel_size=3, strides=1, padding='same',
                              input_shape=(constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION)))
        self.model.add(Activation(activation='relu'))
        self.model.add(MaxPooling2D(pool_size=3, padding='same'))

        # 2. Layer
        self.model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same'))
        self.model.add(Activation(activation='relu'))
        self.model.add(MaxPooling2D(pool_size=3, padding='same'))

        # 3. Layer
        self.model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same'))
        self.model.add(Activation(activation='relu'))
        self.model.add(MaxPooling2D(pool_size=3, padding='same'))

        self.model.add(Dropout(rate=0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))

        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(len(constants.CATEGORIES), activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer,
                      metrics=['accuracy'])

        # display summary of the created model
        self.model.summary()

    def create_model_light(self):
        # give the model a name for tensorboard
        NAME = 'CNN-number-detection-lightnet'
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

        print('[INFO] creating model: ', NAME)

        # create model
        self.model = Sequential()

        # add model layers
        self.model.add(Conv2D(16, kernel_size=3, input_shape=(constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, kernel_size=3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Dropout(rate=0.5))
        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))

        self.model.add(Dense(len(constants.CATEGORIES)))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

        # display summary of the created model
        self.model.summary()

    def create_various_models(self):
        dense_layers = [0, 1, 2]
        layer_sizes = [16, 32, 64]
        conv_layers = [1, 2, 3]

        for dense_layer in dense_layers:
            for layer_size in layer_sizes:
                for conv_layer in conv_layers:
                    NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
                    print('[INFO] creating model: ', NAME)

                    # create model
                    model = Sequential()

                    # add model layers
                    model.add(Conv2D(layer_size,
                                     kernel_size=3,
                                     input_shape=(constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                    for l in range(conv_layer - 1):
                        model.add(Conv2D(layer_size, kernel_size=3))
                        model.add(Activation('relu'))
                        model.add(MaxPooling2D(pool_size=(2, 2)))

                    model.add(Flatten())

                    for _ in range(dense_layer):
                        model.add(Dense(layer_size))
                        model.add(Activation('relu'))

                    model.add(Dense(len(constants.CATEGORIES)))
                    model.add(Activation('softmax'))

                    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

                    model.compile(loss='categorical_crossentropy',
                                  optimizer=self.optimizer,
                                  metrics=['accuracy'],)

                    model.fit(self.trainX, self.trainY,
                              validation_data=(self.testX, self.testY),
                              batch_size=constants.BATCH_SIZE,
                              epochs=constants.EPOCHS,
                              callbacks=[tensorboard])

    def fit_model(self):
        print('[INFO] training model')
        self.model.fit(self.trainX, self.trainY,
                       validation_data=(self.testX, self.testY),
                       batch_size=constants.BATCH_SIZE,
                       epochs=constants.EPOCHS,
                       callbacks=[self.tensorboard])

        print("[INFO] evaluating network")
        predictions = self.model.predict(self.testX, batch_size=32)
        print(classification_report(self.testY.argmax(axis=1),
                                    predictions.argmax(axis=1), target_names=self.lb.classes_))

    def save_model(self):
        print('[INFO] saving model')

        model_path = "{}.h5".format(constants.MODEL_DIR)
        self.model.save(model_path)

        print('[INFO] successfully saved model to: ', model_path)
