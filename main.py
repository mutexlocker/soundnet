import numpy as np
import tensorflow as tf
import librosa
from preprocess import *


feature_dim_2 = 11
save_data_to_array(max_len=feature_dim_2)
X_train, X_test, y_train, y_test = get_train_test()



feature_dim_1 = 20
channel = 1
epochs = 50
batch_size = 100
verbose = 1
num_classes = 3

X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)





def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc




def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)
                                     ,kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.Conv2D(48, kernel_size=(2, 2), activation='relu',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.Conv2D(120, kernel_size=(2, 2), activation='relu',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(64, activation='relu',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax',kernel_initializer='RandomUniform'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer= 'adam',
                  metrics=['accuracy'])
    return model
model = get_model()
model.summary()
model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))
model_json = model.to_json()
with open("model_CNN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_CNN.h5")
print("Saved model to disk")








