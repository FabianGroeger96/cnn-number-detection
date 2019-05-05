INPUT_DATA_DIR = "images_to_extract"
OUTPUT_DATA_DIR = "data_extracted"
CATEGORIES = ["-1", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

MODEL_DIR = "../number_detection_model"

IMG_SIZE = 28
USE_GRAYSCALE = True
DIMENSION = 1

GRADIENT_FROM_RGB = False

SIGNAL_TYPES = ["info", "stop"]

BATCH_SIZE = 16
EPOCHS = 10
VALIDATION_SPLIT = 0.3

# ISOLATOR
# 0: Low, 1: High
CAMERA_POSITION = 0
