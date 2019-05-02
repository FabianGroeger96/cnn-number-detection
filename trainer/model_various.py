import time
import constants
from trainer.model import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard


class ModelVarious(Model):

    def __init__(self, name_postfix='default'):
        super().__init__()

        self.name_postfix = name_postfix

        self.dense_layers = [0, 1, 2]
        self.layer_sizes = [16, 32, 64]
        self.conv_layers = [1, 2, 3]

    def train_model(self):
        for dense_layer in self.dense_layers:
            for layer_size in self.layer_sizes:
                for conv_layer in self.conv_layers:
                    model_name = "{}-conv-{}-nodes-{}-dense-{}-{}".format(conv_layer, layer_size, dense_layer,
                                                                       int(time.time()), self.name_postfix)
                    self.tensorboard = TensorBoard(log_dir="logs/{}".format(model_name))

                    print('[INFO] creating model: ', model_name)

                    # create model
                    self.model = Sequential()

                    # add model layers
                    self.model.add(Conv2D(layer_size,
                                     kernel_size=3,
                                     input_shape=(constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION)))
                    self.model.add(Activation('relu'))
                    self.model.add(MaxPooling2D(pool_size=(2, 2)))

                    for l in range(conv_layer - 1):
                        self.model.add(Conv2D(layer_size, kernel_size=3))
                        self.model.add(Activation('relu'))
                        self.model.add(MaxPooling2D(pool_size=(2, 2)))

                    self.model.add(Flatten())

                    for _ in range(dense_layer):
                        self.model.add(Dense(layer_size))
                        self.model.add(Activation('relu'))

                    self.model.add(Dense(len(constants.CATEGORIES)))
                    self.model.add(Activation('softmax'))

                    self.model.compile(loss='categorical_crossentropy',
                                  optimizer=self.optimizer,
                                  metrics=['accuracy'], )

                    super().train_model()
