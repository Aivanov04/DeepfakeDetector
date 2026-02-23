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

def main():
    print("Hello World!")
    read_data("datasets/Dataset/Train")
    read_data("datasets/Dataset/Test")
    read_data("datasets/Dataset/Validation")

if __name__ == "__main__":
    main()