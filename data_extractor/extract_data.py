from data_extractor.extractor import Extractor


def main():
    print('extracting regions of interest from data')
    extractor = Extractor()
    #extractor.extract_data()

    print('renaming images in categories')
    #extractor.rename_images_in_categories()

    print('creating training data')
    extractor.create_training_data()

    print('creating model')
    extractor.create_model()

if __name__ == "__main__":
    main()
