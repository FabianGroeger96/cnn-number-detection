from data_extractor.extractor import Extractor


def main():
    print('start extracting regions of interest from data')
    extractor = Extractor()
    extractor.extract_data()
    print('finished extracting regions of interest')


if __name__ == "__main__":
    main()
