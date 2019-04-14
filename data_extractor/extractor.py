import os
import cv2
from data_extractor.isolator import Isolator


class Extractor:

    def __init__(self):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        self.DATA_DIR = "/images_to_extract"
        self.CATEGORIES = ["-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        self.isolator = Isolator()

    def set_data_dir(self, data_dir):
        self.DATA_DIR = data_dir

    def set_categories(self, categories):
        self.CATEGORIES = categories

    def extract_data(self):
        current_working_dir = os.getcwd()

        input_dir = current_working_dir + self.DATA_DIR
        print('Input directory: ', input_dir)

        output_dir = current_working_dir + "/data_extracted"
        print('Output directory: ', output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        list_dir = os.listdir(input_dir)
        list_dir = [image for image in list_dir]
        list_dir.sort()

        for image in list_dir:
            image_name = image
            image_name = image_name.partition('.')[0]
            file_string = current_working_dir + self.DATA_DIR + "/" + image

            try:
                image = cv2.imread(file_string)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                regions_of_interest = self.isolator.get_regions_of_interest(image)

                for index, roi in enumerate(regions_of_interest):
                    roi_file_name = output_dir + "/" + image_name + "_" + str(index) + ".jpg"
                    cv2.imwrite(roi_file_name, roi)
            except:
                continue
