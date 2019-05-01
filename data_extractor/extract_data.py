from data_extractor.extractor import Extractor


def main():
    # create an instance of the extractor
    extractor = Extractor()

    # if you want to rename image names
    # extractor.rename_images_in_categories()

    # if you want to extract data from images
    # extractor.extract_data()

    # if you want to categorize images with model
    # extractor.categorize_with_trained_model()

    # if you want to randomly delete images, so that all categories have the same amount of images
    # extractor.randomly_delete_images(200)

    # if you want to generate more data with data augmentation
    # extractor.augment_all_categories()

    # if you need more data, invert image
    # extractor.create_inverse_data('7')

    # if you want to randomly generate images with numpy to filter more false positives
    # extractor.create_random_images('-1', 2000)

    # if you want to create the data model for the cnn
    # extractor.create_training_data()


if __name__ == "__main__":
    main()
