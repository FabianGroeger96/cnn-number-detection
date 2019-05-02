import constants
from model import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard


class ModelGNetLightV2(Model):

    def create_model(self, name_postfix):
        # give the model a name for tensorboard
        model_name = 'CNN-gnet-light-v2-{}'.format(name_postfix)
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(model_name))

        print('[INFO] creating model: ', model_name)

        # create model
        self.model = Sequential()

        # add model layers
        self.model.add(Conv2D(16, kernel_size=3,
                              strides=(2, 2),
                              input_shape=(constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION)))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(32, kernel_size=3, strides=(2, 2)))
        self.model.add(Activation('relu'))

        self.model.add(Dropout(rate=0.5))
        self.model.add(Flatten())

        self.model.add(Dense(len(constants.CATEGORIES)))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

        # display summary of the created model
        self.model.summary()

        # fit (train) the model
        Model.fit_model(self)
