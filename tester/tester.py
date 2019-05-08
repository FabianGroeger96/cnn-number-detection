import cv2
import constants
import os
import shutil
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

            roi_processed = self.isolator.reshape_image_for_input(roi)
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

    def __classify_for_signal(self, image):
        contours = self.isolator.get_contours(image)

        contour_images = []

        for contour_arr in contours:
            contour = contour_arr[0]
            contour_image = contour_arr[1]

            draw_image = np.copy(contour_image)
            if contour is not None:
                cv2.drawContours(draw_image, contour, -1, (255, 153, 255), 1)

            contour_images.append(draw_image)

        return contour_images

    def test_model_with_folder(self, folder_name):
        frames = []

        current_working_dir = os.getcwd()
        listdir = os.path.join(current_working_dir, folder_name)
        listdir = [f for f in listdir]
        listdir.sort()
        for f in listdir:
            file_string = current_working_dir + "/{}/{:s}".format(folder_name, f)
            try:
                frame = cv2.imread(file_string)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                continue
            frames.append(frame)

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        erun_path = os.path.join(current_working_dir, folder_name, 'simulation')
        if os.path.exists(erun_path):
            shutil.rmtree(erun_path, ignore_errors=True)
        os.makedirs(current_working_dir + "/continous/erun/original")
        os.makedirs(current_working_dir + "/continous/erun/contours")

        for i, frame in enumerate(tqdm(frames)):
            result = self.__classify_for_signal(frame)
            frameString = "/pic_{:08d}.jpg".format(i)
            cv2.imwrite((current_working_dir + "/continous/erun/original" + frameString), frame)
            for index, image in enumerate(result):
                frameString = "/pic_{}_{:08d}.jpg".format(index, i)
                cv2.imwrite((current_working_dir + "/continous/erun/contours" + frameString), image)
