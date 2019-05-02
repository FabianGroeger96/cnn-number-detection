import constants
from model import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard


class ModelGNetDeep(Model):

    def create_model(self, name_postfix):
        # give the model a name for tensorboard
        model_name = 'CNN-gnet-deep-{}'.format(name_postfix)
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(model_name))

        print('[INFO] creating model: ', model_name)

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

        # fit (train) the model
        Model.fit_model(self)
