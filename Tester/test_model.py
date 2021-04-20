import argparse
import numpy as np

from Tester.tester import Tester
from Trainer.Models.model_gnet_light import ModelGNetLight
from Trainer.Models.model_gnet_deep import ModelGNetDeep
from Trainer.Models.model_gnet_deep_v2 import ModelGNetDeepV2
from Trainer.Models.model_gnet_deep_deep import ModelGNetDeepDeep

parser = argparse.ArgumentParser(description='cnn-number-detection')
parser.add_argument(
    '--model_type',
    help='type of model to use',
    default='ModelGNetDeep')
parser.add_argument(
    '--model_path', help='path to the saved model',
    default='CNN-gnet-deep-ultimate-data-15-epochs-128-batch-size')
parser.add_argument(
    '--test_image',
    help='path to the image for testing the model')
parser.add_argument('--test_folder', help='folder with images for inference')
parser.add_argument(
    '--test_on_random',
    help='if a randomly generated image should be used for inference',
    action='store_true')
args = parser.parse_args()


def main():
    # create the model to use for inference
    model_class = eval(args.model_type)
    model_obj = model_class('Tester', load_data=False)
    tester = Tester(model_obj, args.model_path)

    if not args.test_image is None:
        # test the model with a given image
        tester.test_model_with_image(args.test_image)

    elif args.test_on_random and model_obj.uses_color:
        # test the model with a random gray scale array
        image_random_gray = np.random.randint(
            0, 255, size=(28, 28), dtype=np.uint8)
        tester.test_model_with_array(image_random_gray)

    elif args.test_on_random and not model_obj.uses_color:
        # test the model with a random color array
        image_random_color = np.random.randint(
            0, 255, size=(28, 28, 3), dtype=np.uint8)
        tester.test_model_with_array(image_random_color)

    elif not args.test_folder is None:
        # test the model with a folder of images
        tester.test_model_with_folder('continuous', display_all=False)


if __name__ == "__main__":
    main()
