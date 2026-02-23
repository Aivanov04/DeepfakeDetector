import tensorflow as tf
from keras.src.utils import image_dataset_from_directory
from tensorflow import keras

# read about __init__.py and __main__.py

# Resizing input images to 299 x 299, default input size for Xception (CNN)
IMG_SIZE = (299, 299)
# Number of images trained at once
BATCH_SIZE = 32

def read_data(directory):
     train_dataset = image_dataset_from_directory(
         directory,
         image_size=IMG_SIZE,
         batch_size=BATCH_SIZE,
         label_mode="binary",
         # Potentially add -> shuffle=True
     )
     print(type(train_dataset))

def build_model():
    model = keras.applications.Xception(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), # Image size, w x h x (#Colour channels)
        include_top=False,                         # Set to false, don't want original ImageNet layer
        weights="imagenet",                        # Keep ImageNet weights, convolutional layers learnt feature detectors
    )

    model.trainable = False

def main():
    print("Hello World!")
    read_data("datasets/Dataset/Train")
    read_data("datasets/Dataset/Test")
    read_data("datasets/Dataset/Validation")

if __name__ == "__main__":
    main()