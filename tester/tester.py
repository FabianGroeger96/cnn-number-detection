import cv2
import constants
from isolator.isolator import Isolator
from trainer.models.model_gnet_light import ModelGNetLight


class Tester:

    def __init__(self):
        model_path = "{}.h5".format(constants.MODEL_DIR)
        model_obj = ModelGNetLight('GNet')
        model_obj.create_model(weights_path=model_path)
        self.model = model_obj.model
        self.model.summary()

        self.isolator = Isolator()

    def test_model_with_image(self, image_path):
        image_array = cv2.imread(image_path)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        regions_of_interest = self.isolator.get_regions_of_interest(image_array)

        for index, roi_arr in enumerate(regions_of_interest):
            roi = roi_arr[0]
            roi_type = roi_arr[1]

            roi_processed = self.isolator.preprocess_image_for_input(roi)
            prediction = self.model.predict([roi_processed])

            i = prediction.argmax(axis=1)[0]
            label = constants.CATEGORIES[i]

            print('Prediction: image type: {}, {} ({:.2f}%)'.format(constants.SIGNAL_TYPES[roi_type],
                                                                    label,
                                                                    prediction[0][i] * 100))

            cv2.imshow("ROI", roi)
            cv2.waitKey(0)

    def test_model_with_array(self, image_array):
        image_processed = self.isolator.reshape_image_for_input(image_array)
        prediction = self.model.predict([image_processed])

        i = prediction.argmax(axis=1)[0]
        label = constants.CATEGORIES[i]

        print('Prediction: {} ({:.2f}%)'.format(label, prediction[0][i] * 100))

        cv2.imshow("ROI", image_array)
        cv2.waitKey(0)
