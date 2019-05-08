import numpy as np
from tester import Tester


def main():
    tester = Tester()

    # test the model with a given image
    # tester.test_model_with_image('frame_overlaying_cnt_2.jpg')

    # test the model with a random color array
    # image_random_color = np.random.randint(0, 255, size=(28, 28, 3), dtype=np.uint8)
    # tester.test_model_with_array(image_random_color)

    # test the model with a random gray scale array
    # image_random_gray = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
    # tester.test_model_with_array(image_random_gray)

    # test the model with a folder of images
    tester.test_model_with_folder('continuous')


if __name__ == "__main__":
    main()
