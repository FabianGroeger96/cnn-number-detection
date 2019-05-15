import pickle
import numpy as np
import constants
import random
import tensorflow as tf
import os
import keras
import matplotlib.cm as cm
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import Adam
from keras import activations
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from vis.visualization import visualize_activation
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam, overlay
from vis.utils import utils
from matplotlib import pyplot as plt

# only show tensorflow errors
tf.logging.set_verbosity(tf.logging.ERROR)
# same for numpy
np.seterr(divide='ignore', invalid='ignore')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Model:

    def __init__(self, model_name):
        print('[INFO] creating model: ', model_name)

        # specify the model
        self.model = None
        self.model_name = model_name

        # specify learning rate for optimizer
        self.optimizer = Adam(lr=1e-3)

        # to start tensorboard run: tensorboard --logdir=logs/, in working directory
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(model_name))

        # load the image data and scale the pixels intensities to range [0, 1]
        X = pickle.load(open("../X.pickle", "rb"))
        X = X / 255

        # load the image data labels
        y = pickle.load(open("../y.pickle", "rb"))

        # train test split
        (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(X, y,
                                                                              test_size=constants.VALIDATION_SPLIT,
                                                                              random_state=42)
        self.lb = LabelBinarizer()
        self.trainY = self.lb.fit_transform(self.trainY)
        self.testY = self.lb.transform(self.testY)

    def create_model(self, weights_path=None):
        raise NotImplementedError

    def save_model(self, visualize_model=False):
        print('[INFO] saving model')
        model_path = "{}{}.h5".format(constants.MODEL_DIR, self.model_name)
        self.model.save(model_path)
        print('[INFO] successfully saved model to: ', model_path)
        if visualize_model:
            self.__visualize_model()

    def train_model(self):
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
        model_output_path = '{}{}.pb'.format(constants.MODEL_DIR, self.model_name)
        output_path_array = model_output_path.split('/')
        output_path = ''
        output_name = ''

        for index, path in enumerate(output_path_array):
            if index != len(output_path_array) - 1:
                output_path = '{}{}/'.format(output_path, path)
            else:
                output_name = path

        keras.backend.set_learning_phase(0)
        model_input_path = "{}{}.h5".format(constants.MODEL_DIR, self.model_name)
        model = keras.models.load_model(model_input_path)

        frozen_graph = self.__freeze_session(keras.backend.get_session(),
                                             output_names=[out.op.name for out in model.outputs])
        tf.train.write_graph(keras.backend.get_session().graph_def, output_path, "graph.pbtxt", as_text=True)
        tf.train.write_graph(frozen_graph, output_path, output_name, as_text=False)
        print('[INFO] successfully saved model to: ', model_output_path)

    def __freeze_session(self, session, keep_var_names=None, output_names=None, clear_devices=True):
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

    def __visualize_model(self):
        self.visualize_model_architecture_image()
        self.visualize_dense_layer()
        self.visualize_feature_map()
        self.visualize_heat_map()

    def visualize_model_architecture_image(self):
        print('[INFO] visualizing model architecture')

        # create folder for saving visualization
        save_path = os.path.join(constants.MODEL_DIR, 'Visualization', self.model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_name = os.path.join(save_path, 'model_architecture_{}.png'.format(self.model_name))
        plot_model(self.model, to_file=file_name, show_shapes=True, show_layer_names=True)

    def visualize_dense_layer(self):
        print('[INFO] visualizing dense layers')

        # create folder for saving visualization
        save_path = os.path.join(constants.MODEL_DIR, 'Visualization', self.model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # search the last dense layer with the name 'preds'
        layer_idx = utils.find_layer_idx(self.model, 'preds')

        # Swap softmax with linear
        self.model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(self.model)

        # output node we want to maximize
        for class_idx in np.arange(len(constants.CATEGORIES)):
            # Lets turn off verbose output this time to avoid clutter and just see the output.
            img = visualize_activation(model, layer_idx, filter_indices=class_idx, input_range=(0., 1.))
            plt.figure()
            plt.title('Networks perception of {}'.format(class_idx))
            plt.imshow(img[..., 0])

            # save the plot
            plot_name = 'dense-layer-{}.png'.format(constants.CATEGORIES[class_idx])
            plt.savefig(os.path.join(save_path, plot_name))
            plt.show()

    def visualize_feature_map(self):
        print('[INFO] visualizing feature map')

        # create folder for saving visualization
        save_path = os.path.join(constants.MODEL_DIR, 'Visualization', self.model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # search the last dense layer with the name 'preds'
        layer_idx = utils.find_layer_idx(self.model, 'preds')

        # Swap softmax with linear
        self.model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(self.model)

        # corresponds to the Dense linear layer
        for class_idx in np.arange(len(constants.CATEGORIES)):
            # choose a random image from test data
            indices = np.where(self.testY[:, class_idx] == 1.)[0]
            idx = random.choice(indices)

            f, ax = plt.subplots(1, 4)
            ax[0].imshow(self.testX[idx][..., 0])

            for i, modifier in enumerate([None, 'guided', 'relu']):
                grads = visualize_saliency(model, layer_idx, filter_indices=class_idx,
                                           seed_input=self.testX[idx], backprop_modifier=modifier)
                if modifier is None:
                    modifier = 'vanilla'

                ax[i + 1].set_title(modifier)
                ax[i + 1].imshow(grads, cmap='jet')

            # save the plot
            plot_name = 'feature-map-{}.png'.format(constants.CATEGORIES[class_idx])
            plt.savefig(os.path.join(save_path, plot_name))
            plt.show()

    def visualize_heat_map(self):
        print('[INFO] visualizing heat map')

        # create folder for saving visualization
        save_path = os.path.join(constants.MODEL_DIR, 'Visualization', self.model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # search the last dense layer with the name 'preds'
        layer_idx = utils.find_layer_idx(self.model, 'preds')

        # Swap softmax with linear
        self.model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(self.model)

        for class_idx in np.arange(len(constants.CATEGORIES)):
            # choose a random image from test data
            indices = np.where(self.testY[:, class_idx] == 1.)[0]
            idx = random.choice(indices)

            f, ax = plt.subplots(1, 4)
            ax[0].imshow(self.testX[idx][..., 0])

            for i, modifier in enumerate([None, 'guided', 'relu']):
                grads = visualize_cam(model, layer_idx, filter_indices=None,
                                      seed_input=self.testX[idx], backprop_modifier=modifier)

                # create heat map to overlay on image
                jet_heat_map = np.uint8(cm.jet(grads)[..., :3] * 255)
                image = np.asarray(self.testX[idx] * 255, np.uint8)
                if constants.USE_GRAY_SCALE:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                if modifier is None:
                    modifier = 'vanilla'

                ax[i + 1].set_title(modifier)
                ax[i + 1].imshow(overlay(jet_heat_map, image))

            # save the plot
            plot_name = 'heat-map-{}.png'.format(constants.CATEGORIES[class_idx])
            plt.savefig(os.path.join(save_path, plot_name))
            plt.show()
