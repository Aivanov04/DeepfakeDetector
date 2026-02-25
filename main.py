import tensorflow as tf
from keras.src.applications.xception import preprocess_input
from keras.src.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.src.utils import image_dataset_from_directory
from tensorflow import keras

# read about __init__.py and __main__.py

# Resizing input images to 299 x 299, default input size for Xception (CNN)
IMG_SIZE = (299, 299)
# Number of images trained at once
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def read_data(directory):
     dataset = image_dataset_from_directory(
         directory,
         image_size=IMG_SIZE,
         batch_size=BATCH_SIZE,
         label_mode="binary",
         # Potentially add -> shuffle=True
     )

     # Xception preprocessing steps
     dataset = dataset.map(
         lambda x, y: (preprocess_input(x), y),
         num_parallel_calls=AUTOTUNE
     )

     return dataset.prefetch(AUTOTUNE)

def build_model():
    basic_model = keras.applications.Xception(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), # Image size, w x h x (#Colour channels)
        include_top=False,                         # Set to false, don't want original ImageNet layer
        weights="imagenet",                        # Keep ImageNet weights, convolutional layers learnt feature detectors
    )

    # Don't update the pretrained weights during new training
    basic_model.trainable = False

    inputs = keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))    # Input parameters
    x = basic_model(inputs)             # Tensor created out of the simple model with inputs layer
    x = GlobalAveragePooling2D()(x)     # Create a layer, flattening to 2D and apply x
    x = Dropout(0.3)(x)                 # Chance to set neurons to 0, prevent overfitting
    outputs = Dense(1, activation="sigmoid")(x) # Dense layer with 1 neuron, to get an output

    model = keras.Model(inputs=inputs, outputs=outputs) # Full model
    # May need adjustments
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def main():
    print("GPUs detected:", tf.config.list_physical_devices('GPU'))
    print("Built with CUDA:", tf.test.is_built_with_cuda())

    train_dataset = read_data("datasets/Dataset/Train")
    val_dataset = read_data("datasets/Dataset/Validation")
    test_dataset = read_data("datasets/Dataset/Test")

    model = build_model()
    model.summary()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10
    )

    print("\nEvaluating on test set:")
    model.evaluate(test_dataset)


if __name__ == "__main__":
    main()