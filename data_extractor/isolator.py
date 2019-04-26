import numpy as np
import cv2
import constants


class Isolator:

    def get_regions_of_interest(self, image):
        regions_of_interest = [] # 0: roi, 1: type

        cropped_images = self._crop(image)
        for index, cropped in enumerate(cropped_images):
            preprocessed_image = self._preprocess(cropped)
            threshold_image = self._threshold(preprocessed_image)
            contours = self._find_contours(threshold_image)
            rois = self._crop_regions_of_interest(cropped, contours)

            for roi in rois:
                roi_arr = []
                roi_arr.append(roi)
                roi_arr.append(index)
                regions_of_interest.append(roi_arr)

        return regions_of_interest

    def get_contours(self, image):
        contours_signal_type = []  # 0: roi, 1: type

        cropped_images = self._crop(image)
        for index, cropped in enumerate(cropped_images):
            preprocessed_image = self._preprocess(cropped)
            threshold_image = self._threshold(preprocessed_image)
            contours = self._find_contours(threshold_image)

            if len(contours) > 0:
                for c in contours:
                    contour_arr = []
                    contour_arr.append(c)
                    contour_arr.append(cropped)
                    contours_signal_type.append(contour_arr)
            else:
                contour_arr = []
                contour_arr.append(None)
                contour_arr.append(cropped)
                contours_signal_type.append(contour_arr)

        return contours_signal_type

    def _crop(self, image):
        image_info = image[100:280, 0:320, :]
        image_stop = image[340:, 0:320, :]

        return [image_info, image_stop]

    def _detect_edges(self, channel):
        sobel_x = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
        sobel_y = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
        sobel = np.hypot(sobel_x, sobel_y)
        sobel[sobel > 255] = 255

        return sobel

    def _preprocess(self, image):
        if constants.GRADIENT_FROM_RGB:
            # create gradient image from all 3 color channels
            # calculate gradient for channels and put it back together
            image = np.max(np.array(
                [self._detect_edges(image[:, :, 0]),
                 self._detect_edges(image[:, :, 1]),
                 self._detect_edges(image[:, :, 2])]), axis=0)
        else:
            # create gradient image from grayscale image
            # for better performance (only 1 channel)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = self._detect_edges(image)
        # calculate mean of the image
        mean = np.mean(image)
        # everything that is below the mean of the image will be set to black
        image[image <= mean + 10] = 0
        # convert the image back to a numpy array
        image = np.asarray(image, np.uint8)

        return image

    def _threshold(self, image):
        image = cv2.inRange(image, 20, 200)

        return image

    def _find_contours(self, image):
        image_height, image_width = image.shape
        contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours_hierarchy = []
        for i, cnt in enumerate(contours):
            if (hierarchy[0][i][3] != -1 and hierarchy[0][i][2] == -1) or \
                    (hierarchy[0][i][3] == -1 and hierarchy[0][i][2] > 0) or \
                    (hierarchy[0][i][3] > 0 and hierarchy[0][i][2] > 0):
                if 370 < cv2.contourArea(cnt) < 2500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w < 55 and h < 100:
                        if 0.35 < w / h < 0.75:
                            length = int(w)
                            height = int(h)

                            point_1_x = int(x + w // 2 - length // 2)
                            point_1_y = int(y + h // 2 - height // 2)
                            point_2_x = point_1_x + length
                            point_2_y = point_1_y + height

                            if point_1_y > 0 and point_1_x > 0 and point_2_y < image_height and point_2_x < image_width:
                                region_of_interest = image[point_1_y:point_2_y, point_1_x:point_2_x]
                                if self._qualifies_as_number(region_of_interest):
                                    contours_hierarchy.append(cnt)

        return contours_hierarchy

    def _qualifies_as_number(self, region_of_interest):
        roi_h, roi_w = region_of_interest.shape
        anz_pixel = roi_h * roi_w
        anz_pixel_white = np.sum(region_of_interest == 255)
        anz_pixel_black = anz_pixel - anz_pixel_white
        anz_pixel_black_ratio = (anz_pixel_black / anz_pixel) * 100

        if anz_pixel_black_ratio <= 35:
            qualifies_as_number = False
        elif anz_pixel_black_ratio >= 75:
            qualifies_as_number = False
        else:
            qualifies_as_number = True

        return qualifies_as_number

    def _crop_regions_of_interest(self, image, contours):
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
                point_2_x

            if point_2_y < 0:
                point_2_y

            region_of_interest = image[point_1_y:point_2_y, point_1_x:point_2_x]
            regions_of_interest.append(region_of_interest)

        return regions_of_interest

