# CNN number detection

## Project

The goal of this repository is to implement a number detection using Tensorflow 
with a custom Convolution Neural Net (CNN) architecture specifically for fast inference.
The CNN will be trained using a custom created dataset that contains numbers from 1-9 
and a category for 'not numbers (-1)'.

*The CNN is fast enough to run in real-time on a Raspberry Pi.*

![Number detection Raspberry Pi](http://fabiangroeger.com/wp-content/uploads/2019/05/cnn-number-detection-gif.gif)

### Specs
- **Image size:** 320 x 240
- **Camera fps:** 38
- **Processing time (with isolator):** 25ms - 45ms (on Raspberry Pi)

### Download links
- [trained models](https://www.dropbox.com/s/hd8yu239te6d0yv/TrainedModels.zip?dl=0)
- [dataset](https://www.dropbox.com/s/iaf18cvt5jaoq06/PREN_dataset.zip?dl=0)
- [example images for isolator](https://www.dropbox.com/s/4ukv5v0r11nnovf/PREN_images.zip?dl=0)

### Includes

- DataExtractor (to extract data for the dataset)
- Isolator (to find the numbers inside an image)
- Trainer (to train the model)
- Tester (to test the model)

## Example input images

![Example input image 1](http://fabiangroeger.com/wp-content/uploads/2019/05/example_input_image_1.jpg)

![Example input image 2](http://fabiangroeger.com/wp-content/uploads/2019/05/example_input_image_2.jpg)

![Example input image 3](http://fabiangroeger.com/wp-content/uploads/2019/05/example_input_image_3.jpg)

## Example extracted images (dataset images)

![Example extracted image 1](http://fabiangroeger.com/wp-content/uploads/2019/05/example_extracted_image_1.jpg)

![Example extracted image 2](http://fabiangroeger.com/wp-content/uploads/2019/05/example_extracted_image_2.jpg)

![Example extracted image 3](http://fabiangroeger.com/wp-content/uploads/2019/05/example_extracted_image_3.jpg)


## Install

### Requirements

You need to have the following packages installed:

- Python 3.6
- Tensorflow 1.4.1+
- OpenCV 4.0
- Etc.

### Install

Clone the repo and install 3rd-party libraries

```bash
$ git clone https://github.com/FabianGroeger96/cnn-number-detection
$ cd cnn-number-detection
$ pip3 install -r requirements.txt
```

## Usage

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

### Label the Data (with existing Model)

1. First of all you will have to extract the regions of interest with the DataExtractor 
(follow the step *Extract data with DataExtractor*)
2. Specify in the `constants.py` file where your model will be located, by modifying the
`MODEL_DIR` constant
3. Place your existing model in the directory that you specified before
4. Run the `extract_data.py` file and call the method `categorize_with_trained_model()`, 
this will categorize your regions of interest
5. Verify that the data was labeled correctly, by checking the `data_extracted` folder

### Create dataset pickle files

1. If you are finished labeling the images, run the `extract_data.py` file and call the method 
`rename_images_in_categories()` from the `Extractor`, this will rename the images 
in each category
2. Run the `extract_data.py` file and call the method `create_training_data()`, 
this will create your pickle files (`X.pickle` and `y.pickle`) which contain 
the data and the labels of your data

### Train the CNN

1. Check if the pickle files (`X.pickle` and `y.pickle`) were created in the root directory
of the project
2. Run the `train_model.py` file within the trainer, this will train your model and save it 
to the directory specified in the `constants.py` (`MODEL_DIR`)

### Test the CNN

1. Check if the model was created in the directory specified in `constants.py` (`MODEL_DIR`)
2. Upload a testing image to the `tester` directory
3. Run the `test_model.py` file within the tester and give the `test_model_with_image(image_name)` 
function the name of the image
4. Run the `test_model.py` file within the tester and give the `test_model_with_folder(folder_name)` 
function the name of the folder containing multiple images to test
