import numpy as np
import tensorflow as tf
import cv2
import io
from PIL import Image
from tensorflow import keras
from tensorflow.python.keras import models
from isolator.isolator import Isolator


class TensorBoardFilterVisualisationCallback(keras.callbacks.Callback):
    def __init__(self, model, name):
        super().__init__()
        self.model = model
        self.name = name
        # Extracts the outputs of the top 7 layers
        self.layer_outputs = [layer.output for layer in self.model.layers[:7]]
        # Creates a model that will return these outputs, given the model input
        self.activation_model = models.Model(inputs=self.model.input,
                                             outputs=self.layer_outputs)

    def on_epoch_end(self, epoch, logs={}):
        images_filters, layer_names = self.__visualize_filters()

        writer = tf.summary.FileWriter('logs/{}'.format(self.name))
        for index, image in enumerate(images_filters):
            layer_name = 'layer_{}_{}'.format(index, layer_names[index])
            image = self.__make_image(image.astype('uint8'))
            summary = tf.Summary(value=[tf.Summary.Value(tag=layer_name, image=image)])
            writer.add_summary(summary, epoch)
        writer.close()

        return

    def __visualize_filters(self):
        isolator = Isolator()
        image_array = cv2.imread('prediction_image.jpg')
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image_processed = isolator.preprocess_image_for_input(image_array)

        activations = self.activation_model.predict([image_processed])

        layer_names = []
        for layer in self.model.layers[:7]:
            layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

        images_per_row = 16

        images_filter_layers = []

        for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
            n_features = layer_activation.shape[-1]  # Number of features in the feature map
            size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
            n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols):  # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0, :, :, col * images_per_row + row]
                    channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                    if channel_image.std() is not 0:
                        channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    # Displays the grid
                    display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
            display_grid = cv2.resize(display_grid, (int(2 * display_grid.shape[1]),
                                                     int(2 * display_grid.shape[0])))
            images_filter_layers.append(display_grid)

        return images_filter_layers, layer_names

    def __make_image(self, image):
        """
        Convert an numpy representation image to Image protobuf.
        Copied from https://github.com/lanpa/tensorboard-pytorch/
        """
        height, width = image.shape
        image = Image.fromarray(image)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=1,
                                encoded_image_string=image_string)
