# CNN number detection

## Project

The goal of this repository is to implement a number detection using Tensorflow 
with a custom Convolution Neural Net (CNN) architecture specifically for fast inference.
The CNN will be trained using a custom created dataset that contains numbers from 1-9 
and a category for 'not numbers'.

*The CNN should be fast enough to run real-time on a Raspberry Pi.*

### Includes

- DataExtractor (to extract data for the dataset)
- Trainer (to train the model)
- Tester (to test the model on custom images)
- Utils (test, simulate the isolator on images)

## Install

### Requirements

You need to have the following packages installed:

- Python 3.6
- Tensorflow 1.1.2
- OpenCV 4.0
- Etc.

### Install

Clone the repo and install 3rd-party libraries

```bash
$ git clone https://github.com/FabianGroeger96/cnn-number-detection
$ cd cnn-number-detection
$ pip3 install -r requirements.txt
```

## Usage (In progress)

### Extract data with DataExtractor

1. Create a folder named `images_to_extract` in the data extractor directory 
(The folder can be named differently, but don't forget to change the `INPUT_DATA_DIR` 
variable in the `constants.py` file).
*This directory will be used to extract the regions of interest to train your CNN.*
2. Copy the data to extract the regions of interest into the `images_to_extract` folder
3. Specify which categories you want to extract in the `constants.py` file, by changing 
the `CATEGORIES` variable
4. Run the `extract_data.py` file and call the method `extract_data()` from the `Extractor`
5. After the method is finished your extracted regions of interest are located in the 
`data_extracted` folder. In there you will also find folders for each of your categories.
These folders are used to label the regions of interest for then training your CNN.

### Label the Data (by Hand)

1. First of all you will have to extract the regions of interest with the DataExtractor 
(follow the step *Extract data with DataExtractor*)
2. Classify the images, by dragging them in the corresponding category folder
3. If you are finished, run the `extract_data.py` file and call the method 
`rename_images_in_categories()` from the `Extractor`, this will rename the images 
in each category
4. Run the `extract_data.py` file and call the method `create_training_data()`, 
this will create your pickle files (`X.pickle` and `y.pickle`) which contain 
the data and the labels of your data