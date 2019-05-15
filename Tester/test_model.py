import numpy as np
from Tester.tester import Tester
from Trainer.Models.model_gnet_light import ModelGNetLight
from Trainer.Models.model_gnet_deep import ModelGNetDeep
from Trainer.Models.model_gnet_deep_v2 import ModelGNetDeepV2
from Trainer.Models.model_gnet_deep_deep import ModelGNetDeepDeep


def main():
    tester = Tester(ModelGNetDeepV2('GNet'), 'CNN-gnet-deep-v2-ultimate-data-15-epochs')

    # test the model with a given image
    # tester.test_model_with_image('frame_overlaying_cnt_2.jpg')

    # test the model with a random color array
    # image_random_color = np.random.randint(0, 255, size=(28, 28, 3), dtype=np.uint8)
    # tester.test_model_with_array(image_random_color)

    # test the model with a random gray scale array
    # image_random_gray = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
    # tester.test_model_with_array(image_random_gray)

    # test the model with a folder of images
    # tester.test_model_with_folder('continuous', display_all=False)

    # create video from images command
    # convert *.jpg recognized.mpeg


if __name__ == "__main__":
    main()
