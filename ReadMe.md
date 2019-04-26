# CNN number detection

## Project

The goal of this repository is to create a custom CNN to detect number from 0-9 on signals.

### Includes

- DataExtractor (to extract data for the dataset)
- Trainer (to train the model)
- Tester (to test the model on custom images)

## Usage

### Extract data with DataExtractor

1. Create a folder named `images_to_extract` in the data extractor directory 
(The folder can be named differently, but don't forget to change the `INPUT_DATA_DIR` 
variable in the `constants.py` file).
*This directory will be used to extract the regions of interest to train your custom CNN.*
2. Copy the data to extract the regions of interest into the `images_to_extract` folder
3. Specify which categories you want to extract in the `constants.py` file, by changing 
the `CATEGORIES` variable
4. Run the `extract_data.py` file and call the method `extract_data`
5. After the method is finished your extracted regions of interest are located in the 
`data_extracted` folder. In there you will also find folders for each of your categories.
These folders are used to label the regions of interest for then training your CNN.

### Label the Data

1. First of all you will have to extract the regions of interest with the DataExtractor
2. Classify the images, by dragging them in the corresponding category folder
3. Run the `extract_data.py` file and call the method `rename_data`