import numpy as np
from tester import Tester


def main():
    tester = Tester()
    # test the model with a given image
    tester.test_model_with_image("frame_number_4.jpg")
    # test the model with a random color array
    tester.test_model_with_array(np.random.rand(28, 28, 3))
    # test the model with a random gray scale array
    tester.test_model_with_array(np.random.rand(28, 28))


if __name__ == "__main__":
    main()
