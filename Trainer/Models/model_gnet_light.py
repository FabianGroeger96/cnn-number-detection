import constants
from Trainer.Models.model import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


class ModelGNetLight(Model):

    def __init__(self, name_postfix):
        # call the init method from superclass
        model_name = 'CNN-gnet-light-{}'.format(name_postfix)
        super().__init__(model_name)

    def create_model(self, weights_path=None):
        # create model
        self.model = Sequential()

        # add model layers
        self.model.add(Conv2D(16, kernel_size=3,
                              input_shape=(constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION),
                              activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Dropout(rate=0.25))  # 0.5 before
        self.model.add(Flatten())

        self.model.add(Dense(len(constants.CATEGORIES), activation='softmax', name='preds'))

        # load weights if path is given
        if weights_path:
            self.model.load_weights(weights_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

        # display summary of the created model
        self.model.summary()
