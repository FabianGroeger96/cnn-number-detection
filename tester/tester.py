import cv2
import constants
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model


class Tester:

    def __init__(self):
        self.model = load_model('../number_detection_model.h5')
        #self.model = load_model('../number_detection_model.model')

    def _preprocess_image(self, image_path):
        image_array = cv2.imread(image_path)
        resized_image_array = cv2.resize(image_array, (constants.IMG_SIZE, constants.IMG_SIZE))
        reshaped_image = resized_image_array.reshape(-1, constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION)

        return reshaped_image

    def test_model(self, image_path):
        image = self._preprocess_image(image_path)
        prediction = self.model.predict([image])

        print(prediction)
