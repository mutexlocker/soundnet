import tensorflow as tf
import numpy as np
from preprocess import *
# Predicts one sample
feature_dim_2 = 11
feature_dim_1 = 20
channel = 1
epochs = 50
batch_size = 100
verbose = 1
num_classes = 3


def predict(filepath, model):
    sample = wav2mfcc(filepath)
    print(sample.shape)
    print(sample)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]


# Model reconstruction from JSON file
with open('model_CNN.json', 'r') as f:
    model = tf.keras.models.model_from_json(f.read())

# Load weights into the new model
model.load_weights('model_CNN.h5')
print(model.summary())
np.set_printoptions(suppress=True)
print(predict('./data/bed/004ae714_nohash_1.wav', model=model))
