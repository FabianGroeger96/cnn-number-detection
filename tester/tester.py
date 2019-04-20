import cv2
import constants
from data_extractor.isolator import Isolator
from g_net import load_model


class Tester:

    def __init__(self):
        model_path = "{}.h5".format(constants.MODEL_DIR)
        self.model = load_model(model_path)
        self.model.summary()

        self.isolator = Isolator()

    def _preprocess_image(self, image_array):
        if constants.USE_GRAYSCALE:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        resized_image_array = cv2.resize(image_array, (constants.IMG_SIZE, constants.IMG_SIZE))
        reshaped_image = resized_image_array.reshape(-1, constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION)

        return reshaped_image

    def test_model(self, image_path):
        image_array = cv2.imread(image_path)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        regions_of_interest, regions_of_interest_type = self.isolator.get_regions_of_interest(image_array)

        for idx, roi_type in enumerate(regions_of_interest):
            for index, roi in enumerate(roi_type):
                image_type = regions_of_interest_type[idx][index]

                roi_processed = self._preprocess_image(roi)
                prediction = self.model.predict([roi_processed])

                i = prediction.argmax(axis=1)[0]
                label = constants.CATEGORIES[i]

                print('Prediction: image type: {}, {} ({:.2f}%)'.format(constants.SIGNAL_TYPES[image_type],
                                                                        label,
                                                                        prediction[0][i] * 100))

                cv2.imshow("ROI", roi)
                cv2.waitKey(0)
