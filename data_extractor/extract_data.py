from time import sleep
from data_extractor.extractor import Extractor


def main():
    # create an instance of the extractor
    extractor = Extractor()

    # if you want to rename image names
    rename_data(extractor)

    # if you want to extract data from images
    #extract_data(extractor)

    # if you want to categorize images with model
    #categorize_data(extractor)

    # if you need more data, invert image
    #extractor.create_inverse_data("7")

    # if you want to create the data model for the cnn
    #create_data_model(extractor)


def extract_data(extractor):
    print('[INFO] extracting regions of interest from data')
    sleep(.5)
    extractor.extract_data()


def rename_data(extractor):
    print('[INFO] renaming images in categories')
    sleep(.5)
    extractor.rename_images_in_categories()


def categorize_data(extractor):
    print('[INFO] categorizing images')
    sleep(.5)
    extractor.categorize_with_trained_model()


def create_data_model(extractor):
    print('[INFO] creating training data')
    sleep(.5)
    extractor.create_training_data()
    sleep(.5)
    print('[INFO] creating data model')
    sleep(.5)
    extractor.create_model()


if __name__ == "__main__":
    main()
