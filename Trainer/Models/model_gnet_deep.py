import constants
from Trainer.Models.model import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


class ModelGNetDeep(Model):

    def __init__(self, name_postfix, load_data=True):
        # call the init method from superclass
        model_name = 'CNN-gnet-deep-{}'.format(name_postfix)
        super().__init__(model_name, load_data=load_data)

    def create_model(self, weights_path=None):
        # create model
        self.model = Sequential()

        # add model layers
        # 1. Layer
        self.model.add(Conv2D(filters=16, kernel_size=3, strides=1, padding='same',
                              input_shape=(constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION),
                              activation='relu'))
        self.model.add(MaxPooling2D(pool_size=3, padding='same'))

        # 2. Layer
        self.model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=3, padding='same'))

        # 3. Layer
        self.model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=3, padding='same'))

        # 4. Layer
        self.model.add(Dropout(rate=0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))

        # 5. Layer
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(len(constants.CATEGORIES), activation='softmax', name='preds'))

        # load weights if path is given
        if weights_path:
            self.model.load_weights(weights_path)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

        # display summary of the created model
        self.model.summary()
