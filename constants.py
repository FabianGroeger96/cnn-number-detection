import logging

INPUT_DATA_DIR = "images_to_extract"
OUTPUT_DATA_DIR = "data_extracted"
CATEGORIES = ["-1", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

MODEL_DIR = "TrainedModels/"

IMG_SIZE = 28
USE_GRAY_SCALE = True
DIMENSION = 1

SIGNAL_TYPES = ["info", "stop"]

BATCH_SIZE = 128
EPOCHS = 15
VALIDATION_SPLIT = 0.25

LOG_LEVEL = logging.INFO
