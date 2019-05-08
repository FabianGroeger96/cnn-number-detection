import numpy as np
import cv2
import constants
from isolator.isolator_constants import IsolatorConstants
from isolator.isolator_constants_320_240 import IsolatorConstants320240
from isolator.isolator_constants_640_480 import IsolatorConstants640480


class Isolator:

    CLASS_CONSTANTS = IsolatorConstants

    def get_regions_of_interest(self, image):
        # 0: roi, 1: type
        regions_of_interest = []

        if self.CLASS_CONSTANTS is not None:
            self.__set_constants(image)

        cropped_images = self.__crop(image)
        for index, cropped in enumerate(cropped_images):
            preprocessed_image = self.__preprocess(cropped)
            threshold_image = self.__threshold(preprocessed_image)
            contours = self.__find_contours(threshold_image)
            rois = self.__crop_regions_of_interest(cropped, contours)

            for roi in rois:
                roi_arr = [roi, index]
                regions_of_interest.append(roi_arr)

        return regions_of_interest

    def get_contours(self, image):
        # 0: roi, 1: type
        contours_signal_type = []

        if self.CLASS_CONSTANTS is not None:
            self.__set_constants(image)

        cropped_images = self.__crop(image)
        for index, cropped in enumerate(cropped_images):
            preprocessed_image = self.__preprocess(cropped)
            threshold_image = self.__threshold(preprocessed_image)
            contours = self.__find_contours(threshold_image)

            if len(contours) > 0:
                contour_arr = [contours, cropped]
                contours_signal_type.append(contour_arr)
            else:
                contour_arr = [None, cropped]
                contours_signal_type.append(contour_arr)

        return contours_signal_type

    def __set_constants(self, image):
        image_height, image_width, _ = image.shape

        if image_height == 240 and image_width == 320:
            self.CONSTANTS = IsolatorConstants320240
        elif image_height == 480 and image_width == 640:
            self.CONSTANTS = IsolatorConstants640480

    def __crop(self, image):
        info_start = self.CONSTANTS.CROP_INFO_HEIGHT_START
        info_end = self.CONSTANTS.CROP_INFO_HEIGHT_END

        stop_start = self.CONSTANTS.CROP_STOP_HEIGHT_START
        stop_end = self.CONSTANTS.CROP_STOP_HEIGHT_END

        width_start = self.CONSTANTS.CROP_WIDTH_START
        width_end = self.CONSTANTS.CROP_WIDTH_END

        image_info = image[info_start:info_end, width_start:width_end, :]
        image_stop = image[stop_start:stop_end, width_start:width_end, :]

        return [image_info, image_stop]

    def __detect_edges(self, channel):
        sobel_x = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
        sobel_y = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
        sobel = np.hypot(sobel_x, sobel_y)
        sobel[sobel > 255] = 255

        return sobel

    def __preprocess(self, image):
        if constants.GRADIENT_FROM_RGB:
            # create gradient image from all 3 color channels
            # calculate gradient for channels and put it back together
            image = np.max(np.array(
                [self.__detect_edges(image[:, :, 0]),
                 self.__detect_edges(image[:, :, 1]),
                 self.__detect_edges(image[:, :, 2])]), axis=0)
        else:
            # create gradient image from gray scale image
            # for better performance (only 1 channel)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = self.__detect_edges(image)
        # calculate mean of the image
        mean = np.mean(image)
        # everything that is below the mean of the image will be set to black
        image[image <= mean + self.CONSTANTS.ADDITION_MEAN] = 0
        # convert the image back to a numpy array
        image = np.asarray(image, np.uint8)

        return image

    def __threshold(self, image):
        image = cv2.inRange(image, self.CONSTANTS.THRESHOLD_LOWER, self.CONSTANTS.THRESHOLD_UPPER)

        return image

    def __find_contours(self, image):
        image_height, image_width = image.shape
        _, contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours_hierarchy = []
        for i, cnt in enumerate(contours):
            if (hierarchy[0][i][3] != -1 and hierarchy[0][i][2] == -1) or \
                    (hierarchy[0][i][3] == -1 and hierarchy[0][i][2] > 0) or \
                    (hierarchy[0][i][3] > 0 and hierarchy[0][i][2] > 0):
                if self.CONSTANTS.AREA_SIZE_MIN < cv2.contourArea(cnt) < self.CONSTANTS.AREA_SIZE_MAX:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w < self.CONSTANTS.WIDTH_MAX and h < self.CONSTANTS.HEIGHT_MAX:
                        if self.CONSTANTS.WIDTH_HEIGHT_RATIO_MIN < w / h < self.CONSTANTS.WIDTH_HEIGHT_RATIO_MAX:
                            length = int(w)
                            height = int(h)

                            point_1_x = int(x + w // 2 - length // 2)
                            point_1_y = int(y + h // 2 - height // 2)
                            point_2_x = point_1_x + length
                            point_2_y = point_1_y + height

                            if point_1_y > 0 and point_1_x > 0 and point_2_y < image_height and point_2_x < image_width:
                                region_of_interest = image[point_1_y:point_2_y, point_1_x:point_2_x]
                                if self.__qualifies_as_number(region_of_interest):
                                    contours_hierarchy.append(cnt)

        return contours_hierarchy

    def __qualifies_as_number(self, region_of_interest):
        roi_h, roi_w = region_of_interest.shape
        anz_pixel = roi_h * roi_w
        anz_pixel_white = np.sum(region_of_interest == 255)
        anz_pixel_black = anz_pixel - anz_pixel_white
        anz_pixel_black_ratio = (anz_pixel_black / anz_pixel) * 100

        if anz_pixel_black_ratio <= self.CONSTANTS.PIXEL_RATIO_MIN:
            qualifies_as_number = False
        elif anz_pixel_black_ratio >= self.CONSTANTS.PIXEL_RATIO_MAX:
            qualifies_as_number = False
        else:
            qualifies_as_number = True

        return qualifies_as_number

    def __crop_regions_of_interest(self, image, contours):
        regions_of_interest = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            length = int(w * 1.1)
            height = int(h * 1.1)

            point_1_x = int(x + w // 2 - length // 2)
            point_1_y = int(y + h // 2 - height // 2)
            point_2_x = point_1_x + length
            point_2_y = point_1_y + height

            if point_1_x < 0:
                point_1_x = 0
            if point_1_y < 0:
                point_1_y = 0
            if point_2_x < 0:
                point_2_x = 0
            if point_2_y < 0:
                point_2_y = 0

            region_of_interest = image[point_1_y:point_2_y, point_1_x:point_2_x]
            regions_of_interest.append(region_of_interest)

        return regions_of_interest

    def reshape_image_for_input(self, image_array):
        resized_image_array = cv2.resize(image_array, (constants.IMG_SIZE, constants.IMG_SIZE))
        reshaped_image = resized_image_array.reshape(-1, constants.IMG_SIZE, constants.IMG_SIZE, constants.DIMENSION)

        return reshaped_image

    def preprocess_image_for_input(self, image_array):
        if constants.USE_GRAYSCALE:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        reshaped_image = self.reshape_image_for_input(image_array)

        return reshaped_image
