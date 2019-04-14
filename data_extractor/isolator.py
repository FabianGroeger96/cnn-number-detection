import numpy as np
import cv2


class Isolator:

    def get_regions_of_interest(self, image):
        cropped_image = self._crop(image)
        preprocessed_image = self._preprocess(cropped_image)
        threshold_image = self._threshold(preprocessed_image)
        contours = self._find_contours(threshold_image)
        regions_of_interest = self._crop_regions_of_interest(cropped_image, contours)

        return regions_of_interest

    def _crop(self, image):
        image = image[100:280, 0:320, :]

        return image

    def _detect_edges(self, channel):
        sobel_x = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
        sobel_y = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
        sobel = np.hypot(sobel_x, sobel_y)
        sobel[sobel > 255] = 255

        return sobel

    def _preprocess(self, image):
        # create gradient image from all 3 color channels
        # calculate gradient for channels and put it back together
        image = np.max(np.array(
            [self._detect_edges(image[:, :, 0]),
             self._detect_edges(image[:, :, 1]),
             self._detect_edges(image[:, :, 2])]), axis=0)
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
        contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours_hierarchy = []
        for i, cnt in enumerate(contours):
            if (hierarchy[0][i][3] != -1 and hierarchy[0][i][2] == -1) or \
                    (hierarchy[0][i][3] == -1 and hierarchy[0][i][2] > 0) or \
                    (hierarchy[0][i][3] > 0 and hierarchy[0][i][2] > 0):
                if 450 < cv2.contourArea(cnt) < 2500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w < 55 and h < 100:
                        if 0.3 < w / h < 0.75:
                            length = int(w)
                            height = int(h)

                            point_1 = int(y + h // 2 - height // 2)
                            point_2 = int(x + w // 2 - length // 2)

                            region_of_interest = image[point_1:point_1 + height, point_2:point_2 + length]

                            if self._qualifies_as_number(region_of_interest):
                                box_classified = True
                                rect = cv2.minAreaRect(cnt)
                                box = cv2.boxPoints(rect)
                                box = np.int0(box)
                                for point in box:
                                    if point[0] == 0 or point[1] == 0:
                                        box_classified = False
                                if box_classified:
                                    contours_hierarchy.append(cnt)
        return contours_hierarchy

    def _qualifies_as_number(self, region_of_interest):
        roi_h, roi_w = region_of_interest.shape
        anz_pixel = roi_h * roi_w
        anz_pixel_white = np.sum(region_of_interest == 255)
        anz_pixel_black = anz_pixel - anz_pixel_white
        anz_pixel_black_ratio = (anz_pixel_black / anz_pixel) * 100

        if anz_pixel_black_ratio <= 25:
            qualifies_as_number = False
        elif anz_pixel_black_ratio >= 50:
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

            point_1 = int(y + h // 2 - height // 2)
            point_2 = int(x + w // 2 - length // 2)

            region_of_interest = image[point_1:point_1 + height, point_2:point_2 + length]
            regions_of_interest.append(region_of_interest)

        return regions_of_interest

