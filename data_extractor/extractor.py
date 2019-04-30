import os
import time

import cv2
import random
import numpy as np
import pickle
import constants
import random

from natsort import natsorted
from tqdm import tqdm
from utils.isolator.isolator import Isolator
from tester.g_net import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import array_to_img, img_to_array, load_img


class Extractor:

    def __init__(self):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        self.image_data_gen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=[0.8, 1.1],
            brightness_range=[0.5, 1.5],
            fill_mode='reflect')

        self.isolator = Isolator()
        self.current_working_dir = os.getcwd()

        self.training_data = []

    def extract_data(self):
        print('[INFO] extracting regions of interest from data')

        input_dir = os.path.join(self.current_working_dir, constants.INPUT_DATA_DIR)
        print('[INFO] Input directory: ', input_dir)

        output_dir = os.path.join(self.current_working_dir, constants.OUTPUT_DATA_DIR)
        print('[INFO] Output directory: ', output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        list_dir = os.listdir(input_dir)
        list_dir = natsorted(list_dir)

        for image in tqdm(list_dir):
            image_name = image
            image_name = image_name.partition('.')[0]
            file_string = self.current_working_dir + "/{:s}/{:s}".format(constants.INPUT_DATA_DIR, image)

            try:
                image = cv2.imread(file_string)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    regions_of_interest = self.isolator.get_regions_of_interest(image)

                    for index, roi_arr in enumerate(regions_of_interest):
                        roi = roi_arr[0]
                        roi_type = roi_arr[1]
                        roi_file_name = output_dir + "/{:s}_{:s}_{:s}.jpg".format(image_name, str(index), str(roi_type))
                        cv2.imwrite(roi_file_name, roi)

            except Exception as e:
                print(e)

        print('[INFO] creating folders for sorting rois in categories')
        for category in constants.CATEGORIES:
            category_dir = os.path.join(output_dir, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)

    def rename_images_in_categories(self):
        print('[INFO] renaming images in categories')
        for category in constants.CATEGORIES:
            category_dir = os.path.join(self.current_working_dir, constants.OUTPUT_DATA_DIR, category)

            list_category_dir = os.listdir(category_dir)
            list_category_dir = natsorted(list_category_dir)

            for index, image in enumerate(tqdm(list_category_dir)):
                image_path = os.path.join(category_dir, image)
                image_array = cv2.imread(image_path)

                if image_array is not None:
                    # remove the loaded image
                    os.remove(image_path)

                    image_name = "{:s}.jpg".format(str(index))
                    image_path = os.path.join(category_dir, image_name)
                    if not os.path.exists(image_path):
                        cv2.imwrite(image_path, image_array)
                    else:
                        i = index
                        while True:
                            i += 1
                            image_name = "{:s}.jpg".format(str(i))
                            image_path = os.path.join(category_dir, image_name)
                            if not os.path.exists(image_path):
                                cv2.imwrite(image_path, image_array)
                                break

    def create_inverse_data(self, category):
        print('[INFO] creating inverse data in category {}'.format(category))

        category_dir = os.path.join(self.current_working_dir, constants.OUTPUT_DATA_DIR, category)

        list_category_dir = os.listdir(category_dir)
        list_category_dir = natsorted(list_category_dir)

        for index, image in enumerate(tqdm(list_category_dir)):
            image_path = os.path.join(category_dir, image)
            image_array = cv2.imread(image_path)

            if image_array is not None:
                # invert image
                image_inv = cv2.bitwise_not(image_array)

                image_name = "{:s}_inv.jpg".format(str(index))
                image_path = os.path.join(category_dir, image_name)
                cv2.imwrite(image_path, image_inv)

    def create_training_data(self):
        print('[INFO] creating training data')

        self.training_data.clear()
        for category in constants.CATEGORIES:

            category_dir = os.path.join(self.current_working_dir, constants.OUTPUT_DATA_DIR, category)

            list_category_dir = os.listdir(category_dir)
            list_category_dir = natsorted(list_category_dir)

            for img in tqdm(list_category_dir):
                try:
                    img_array = cv2.imread(os.path.join(category_dir, img))
                    # convert image to grayscale if parameter is set in constants file
                    if constants.USE_GRAYSCALE:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    # resize to normalize data size
                    new_array = cv2.resize(img_array, (constants.IMG_SIZE, constants.IMG_SIZE))
                    # add image to our training data
                    self.training_data.append([new_array, category])
                # exceptions are ignored, to keep the output clean
                except Exception as e:
                    pass

        random.shuffle(self.training_data)

        self._create_model()

    def _create_model(self):
        print('[INFO] creating data model')

        X = []
        y = []

        for features, label in self.training_data:
            X.append(features)
            y.append(label)

        X = np.array(X).reshape(-1, constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION)

        pickle_out = open("../X.pickle", "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("../y.pickle", "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

        print('[INFO] saved data model to root directory')

    def categorize_with_trained_model(self):
        print('[INFO] categorizing images')

        model_path = "{}.h5".format(constants.MODEL_DIR)
        model = load_model(model_path)

        extracted_data_dir = os.path.join(self.current_working_dir, constants.OUTPUT_DATA_DIR)

        list_data_dir = os.listdir(extracted_data_dir)
        list_data_dir = natsorted(list_data_dir)

        for img in tqdm(list_data_dir):
            try:
                image_path = os.path.join(extracted_data_dir, img)
                img_array = cv2.imread(image_path)
                if img_array is not None:
                    if constants.USE_GRAYSCALE:
                        image_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    resized_image_array = cv2.resize(image_array, (constants.IMG_SIZE, constants.IMG_SIZE))
                    reshaped_image = resized_image_array.reshape(-1, constants.IMG_SIZE, constants.IMG_SIZE,
                                                                 constants.DIMENSION)
                    prediction = model.predict([reshaped_image])

                    i = prediction.argmax(axis=1)[0]
                    label = constants.CATEGORIES[i]

                    if prediction[0][i] > 0.95:
                        os.remove(image_path)
                        image_path = os.path.join(extracted_data_dir, label, img)
                        cv2.imwrite(image_path, image_array)

            # exceptions are ignored, to keep the output clean
            except Exception as e:
                pass

    def randomly_delete_images(self, count_files_after_delete):
        print('[INFO] randomly deleting images from categories')

        for category in constants.CATEGORIES:
            category_dir = os.path.join(self.current_working_dir, constants.OUTPUT_DATA_DIR, category)

            list_category_dir = os.listdir(category_dir)
            count_delete = len(list_category_dir) - count_files_after_delete

            if count_delete > 0:

                print('[INFO] deleting {} images in category {}'.format(count_delete, category))

                for i in range(0, count_delete - 1):
                    while True:
                        chosen_image = random.choice(list_category_dir)

                        image_path = os.path.join(category_dir, chosen_image)
                        image_array = cv2.imread(image_path)

                        if image_array is not None:
                            # remove the loaded image
                            os.remove(image_path)
                            break

        self.rename_images_in_categories()

    def augment_all_categories(self):
        print('[INFO] generating data')
        # loop through all categories
        for category in constants.CATEGORIES:
            self.augment_category(category)

    def augment_category(self, category):
        time.sleep(.5)
        print('[INFO] augmenting images in category {}'.format(category))
        time.sleep(.5)
        # create the input path for category
        category_dir = os.path.join(self.current_working_dir, constants.OUTPUT_DATA_DIR, category)
        # searches files in category dir
        list_category_dir = os.listdir(category_dir)
        list_category_dir = natsorted(list_category_dir)
        # loops through all files in path
        for img in tqdm(list_category_dir):
            if '.jpg' in img or '.png' in img:
                image_path = os.path.join(category_dir, img)
                # load image to array
                image = load_img(image_path)
                if image is not None:
                    # convert image to array
                    image = img_to_array(image)
                    # reshape to array rank 4
                    image = image.reshape((1,) + image.shape)
                    # create infinite flow of images
                    images_flow = self.image_data_gen.flow(image, batch_size=1)
                    for i, new_images in enumerate(images_flow):
                        # we access only first image because of batch_size=1
                        new_image = array_to_img(new_images[0], scale=True)
                        # save the augmented image
                        image_name = "{:s}_{:s}_aug.jpg".format(img, str(i))
                        image_path = os.path.join(category_dir, image_name)
                        new_image.save(image_path)
                        # break infinite loop after generated 10 images
                        if i >= 10:
                            break
