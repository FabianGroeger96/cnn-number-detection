import os
import cv2
import random
import numpy as np
import pickle

from tqdm import tqdm
from data_extractor.isolator import Isolator


class Extractor:

    def __init__(self):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        self.DATA_DIR = "/images_to_extract"
        self.CATEGORIES = ["-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.IMG_SIZE = 25

        self.isolator = Isolator()
        self.current_working_dir = os.getcwd()

        self.training_data = []

    def set_data_dir(self, data_dir):
        self.DATA_DIR = data_dir

    def set_categories(self, categories):
        self.CATEGORIES = categories

    def set_image_size(self, image_size):
        self.IMG_SIZE = image_size

    def extract_data(self):
        input_dir = self.current_working_dir + self.DATA_DIR
        print('Input directory: ', input_dir)

        output_dir = self.current_working_dir + "/data_extracted"
        print('Output directory: ', output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        list_dir = os.listdir(input_dir)
        list_dir = [image for image in list_dir]
        list_dir.sort()

        for image in tqdm(list_dir):
            image_name = image
            image_name = image_name.partition('.')[0]
            file_string = self.current_working_dir + "{:s}/{:s}".format(self.DATA_DIR, image)

            try:
                image = cv2.imread(file_string)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                regions_of_interest = self.isolator.get_regions_of_interest(image)

                for index, roi in enumerate(regions_of_interest):
                    roi_file_name = output_dir + "/{:s}_{:s}.jpg".format(image_name, str(index))
                    cv2.imwrite(roi_file_name, roi)
            except:
                continue

        print('creating folders for sorting rois in categories')
        for category in self.CATEGORIES:
            category_dir = output_dir + "/{:s}".format(category)
            if not os.path.exists(category):
                os.makedirs(category_dir)

    def rename_images_in_categories(self):
        for category in self.CATEGORIES:
            category_dir = os.path.join(self.current_working_dir, "data_extracted", category)
            for index, image in enumerate(os.listdir(category_dir)):
                image_array = cv2.imread(os.path.join(category_dir, image))
                image_name = "{:s}.jpg".format(str(index))
                image_path = os.path.join(category_dir, image_name)
                cv2.imwrite(image_path, image_array)

                # if the name and new name are the same, dont remove image
                # because otherwise this image would be deleted
                if image_name != image:
                    image_path = os.path.join(category_dir, image)
                    os.remove(image_path)

    def create_training_data(self):
        self.training_data.clear()
        for category in self.CATEGORIES:

            category_dir = os.path.join(self.current_working_dir, "data_extracted", category)
            class_num = self.CATEGORIES.index(category)

            for img in tqdm(os.listdir(category_dir)):
                try:
                    img_array = cv2.imread(os.path.join(category_dir, img))
                    # resize to normalize data size
                    new_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
                    # add image to our training data
                    self.training_data.append([new_array, class_num])

                except Exception as e:
                    pass
                #except OSError as e:
                #    print("OSErrorBad image most likely broken", e, os.path.join(path, img))
                #except Exception as e:
                #    print("general exception", e, os.path.join(category_dir, img))

        random.shuffle(self.training_data)

    def create_model(self):
        X = []
        y = []

        for features, label in self.training_data:
            X.append(features)
            y.append(label)

        X = np.array(X).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3)

        pickle_out = open("data_extracted/X.pickle", "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("data_extracted/y.pickle", "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

