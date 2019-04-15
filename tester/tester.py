import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model


class Tester:

    def __init__(self):
        self.CATEGORIES = ["-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.IMG_SIZE = 28

        self.model = load_model('../number_detection_model.h5')
        #self.model = load_model('../number_detection_model.model')

    def _preprocess_image(self, image_path):
        image_array = cv2.imread(image_path)
        resized_image_array = cv2.resize(image_array, (self.IMG_SIZE, self.IMG_SIZE))
        reshaped_image = resized_image_array.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3)

        return reshaped_image

    def test_model(self, image_path):
        image = self._preprocess_image(image_path)
        prediction = self.model.predict([image])

        print(prediction)
