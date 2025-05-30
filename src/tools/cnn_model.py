from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from config import MODEL_DIR, SYMBOLS
from tools.image_input import read_img_file
import tensorflow as tf
import numpy as np
import logging  # Import Python's logging module

# Set up logging
logging.basicConfig(level=logging.INFO)

def create_cnn_model():
    """Create CNN model using Keras."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(45, 45, 1), padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=3),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='softmax')
    ])
    return model

# Create the model
cnn_symbol_classifier = create_cnn_model()
cnn_symbol_classifier.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])

# CNN classifier training function
def train_cnn_model(steps):
    train_data, train_data_labels = read_img_file('train')

    # Train the model
    cnn_symbol_classifier.fit(train_data, train_data_labels, epochs=steps, batch_size=100, shuffle=True)

def eval_cnn_model():
    eval_data, eval_data_labels, filelist = read_img_file('eval')
    # Evaluate the model and print results
    eval_results = cnn_symbol_classifier.evaluate(eval_data, eval_data_labels)
    print(eval_results)

if __name__ == "__main__":
    steps = 10  # Adjust the number of epochs as needed
    train_cnn_model(steps)
    eval_cnn_model()
