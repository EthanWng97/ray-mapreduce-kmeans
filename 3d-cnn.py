from __future__ import division, print_function, absolute_import
from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPool3D, BatchNormalization, Input
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, TensorBoard
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


# Hyper Parameter
batch_size = 86
epochs = 20

# Set up TensorBoard
tensorboard = TensorBoard(batch_size=batch_size)

with h5py.File("/Users/wangyifan/Google Drive/3dmnist/full_dataset_vectors.h5", 'r') as h5:
    X_train, y_train = h5["X_train"][:], h5["y_train"][:]
    X_test, y_test = h5["X_test"][:], h5["y_test"][:]

# Translate data to color


def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:, :-1]


def translate(x):
    xx = np.ndarray((x.shape[0], 4096, 3))
    for i in range(x.shape[0]):
        xx[i] = array_to_color(x[i])
        if i % 1000 == 0:
            print(i)
    # Free Memory
    del x

    return xx


y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)

X_train = translate(X_train).reshape(-1, 16, 16, 16, 3)
X_test = translate(X_test).reshape(-1, 16, 16, 16, 3)

# Conv3D layer


def Conv(filters=16, kernel_size=(3, 3, 3), activation='relu', input_shape=None):
    if input_shape:
        return Conv3D(filters=filters, kernel_size=kernel_size, padding='Same', activation=activation, input_shape=input_shape)
    else:
        return Conv3D(filters=filters, kernel_size=kernel_size, padding='Same', activation=activation)

# Define Model


def CNN(input_dim, num_classes):
    model = Sequential()

    model.add(Conv(8, (3, 3, 3), input_shape=input_dim))
    model.add(Conv(16, (3, 3, 3)))
    # model.add(BatchNormalization())
    model.add(MaxPool3D())
    # model.add(Dropout(0.25))

    model.add(Conv(32, (3, 3, 3)))
    model.add(Conv(64, (3, 3, 3)))
    model.add(BatchNormalization())
    model.add(MaxPool3D())
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    return model

# Train Model


def train(optimizer, scheduler):
    global model

    print("Training...")
    model.compile(optimizer='adam',
                  loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.15,
              verbose=2, callbacks=[scheduler, tensorboard])


def evaluate():
    global model

    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)

    print(accuracy_score(pred, y_test))
    # Heat Map
    array = confusion_matrix(y_test, pred)
    cm = pd.DataFrame(array, index=range(10), columns=range(10))
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True)
    plt.show()


def save_model():
    global model

    model_json = model.to_json()
    with open('model/model_3D.json', 'w') as f:
        f.write(model_json)

    model.save_weights('model/model_3D.h5')

    print('Model Saved.')


def load_model():
    f = open('model/model_3D.json', 'r')
    model_json = f.read()
    f.close()

    loaded_model = model_from_json(model_json)
    loaded_model.load_weights('model/model_3D.h5')

    print("Model Loaded.")
    return loaded_model


if __name__ == '__main__':

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    scheduler = ReduceLROnPlateau(
        monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=1e-5)

    model = CNN((16, 16, 16, 3), 10)

    train(optimizer, scheduler)
    evaluate()
    save_model()
