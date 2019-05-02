import os
import pickle
import time

import constants
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard

# disable tensorflow debugging (INFO, WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model:

    def __init__(self):
        # specify the model
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

    def save_model(self):
        print('[INFO] saving model')
        model_path = "{}.h5".format(constants.MODEL_DIR)
        self.model.save(model_path)
        print('[INFO] successfully saved model to: ', model_path)

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

    def convert_model_tensorflow(self):
        print('[INFO] saving model for tensorflow')
        model_output_path = '{}.pb'.format(constants.MODEL_DIR)
        output_path_array = model_output_path.split('/')
        output_path = ''
        output_name = ''

        for index, path in enumerate(output_path_array):
            if index != len(output_path_array) - 1:
                output_path = '{}{}/'.format(output_path, path)
            else:
                output_name = path

        keras.backend.set_learning_phase(0)
        model_input_path = "{}.h5".format(constants.MODEL_DIR)
        model = keras.models.load_model(model_input_path)

        frozen_graph = self.__convert_keras_to_tensorflow(keras.backend.get_session(),
                                                          output_names=[out.op.name for out in model.outputs])
        tf.train.write_graph(keras.backend.get_session().graph_def, output_path, "graph.pbtxt", as_text=True)
        tf.train.write_graph(frozen_graph, output_path, output_name, as_text=False)
        print('[INFO] successfully saved model to: ', model_output_path)

    def __convert_keras_to_tensorflow(self, session, keep_var_names=None, output_names=None, clear_devices=True):
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph

    def create_model(self, name_postfix):
        raise NotImplementedError
