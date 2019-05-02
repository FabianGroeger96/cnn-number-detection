import constants
from trainer.models.model import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard


class ModelGNetLight(Model):

    def __init__(self, name_postfix='default', weights_path=None):
        super().__init__()

        # give the model a name for tensorboard
        self.model_name = 'CNN-gnet-light-{}'.format(name_postfix)
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(self.model_name))

        print('[INFO] creating model: ', self.model_name)

        # create model
        self.model = Sequential()

        # add model layers
        self.model.add(
            Conv2D(16, kernel_size=3, input_shape=(constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, kernel_size=3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Dropout(rate=0.5))
        self.model.add(Flatten())

        self.model.add(Dense(len(constants.CATEGORIES)))
        self.model.add(Activation('softmax'))

        # load weights if path is given
        if weights_path:
            self.model.load_weights(weights_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

        # display summary of the created model
        self.model.summary()
