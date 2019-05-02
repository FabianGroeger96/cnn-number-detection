import cv2
import constants
from utils.isolator.isolator import Isolator
from trainer.model_gnet_light import ModelGNetLight


class Tester:

    def __init__(self):
        model_path = "{}.h5".format(constants.MODEL_DIR)
        model_obj = ModelGNetLight(weights_path=model_path)
        self.model = model_obj.get_model()
        self.model.summary()

        self.isolator = Isolator()

    def __reshape_image(self, image_array):
        resized_image_array = cv2.resize(image_array, (constants.IMG_SIZE, constants.IMG_SIZE))
        reshaped_image = resized_image_array.reshape(-1, constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION)

        return reshaped_image

    def __preprocess_image(self, image_array):
        if constants.USE_GRAYSCALE:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        reshaped_image = self.__reshape_image(image_array)

        return reshaped_image

    def test_model_with_image(self, image_path):
        image_array = cv2.imread(image_path)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        regions_of_interest = self.isolator.get_regions_of_interest(image_array)

        for index, roi_arr in enumerate(regions_of_interest):
            roi = roi_arr[0]
            roi_type = roi_arr[1]

            roi_processed = self.__preprocess_image(roi)
            prediction = self.model.predict([roi_processed])

            i = prediction.argmax(axis=1)[0]
            label = constants.CATEGORIES[i]

            print('Prediction: image type: {}, {} ({:.2f}%)'.format(constants.SIGNAL_TYPES[roi_type],
                                                                    label,
                                                                    prediction[0][i] * 100))

            cv2.imshow("ROI", roi)
            cv2.waitKey(0)

    def test_model_with_array(self, image_array):
        image_processed = self.__reshape_image(image_array)
        prediction = self.model.predict([image_processed])

        i = prediction.argmax(axis=1)[0]
        label = constants.CATEGORIES[i]

        print('Prediction: {} ({:.2f}%)'.format(label, prediction[0][i] * 100))

        cv2.imshow("ROI", image_array)
        cv2.waitKey(0)
