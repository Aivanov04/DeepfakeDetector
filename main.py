import tensorflow as tf
from keras.src.utils import image_dataset_from_directory
from tensorflow import keras

# read about __init__.py and __main__.py

def read_data():
     train_dataset = image_dataset_from_directory(
         "datasets/Dataset/Train"
     )
     print(train_dataset)

def main():
    print("Hello World!")
    read_data()

if __name__ == "__main__":
    main()