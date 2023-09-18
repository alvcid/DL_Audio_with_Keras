import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras import models
from tensorflow.keras import layers

# 0. Download the data
DATASET_ROOT = r"C:\Users\alvar\Documents\Udemy\2_Curso_ Aprende Inteligencia Artificial y Deep Learning con Python &\7. Clasificación+de+audio+con+el+Perceptrón+Multicapa+I.ipynb\16000_pcm_speeches\16000_pcm_speeches"
BENJAMIN_DATA = os.path.join(DATASET_ROOT, "Benjamin_Netanyau")
JENS_DATA = os.path.join(DATASET_ROOT, "Jens_Stoltenberg")
JULIA_DATA = os.path.join(DATASET_ROOT, "Julia_Gillard")
MARGARET_DATA = os.path.join(DATASET_ROOT, "Magaret_Tarcher")
NELSON_DATA = os.path.join(DATASET_ROOT, "Nelson_Mandela")

# 1. Data preparation
def parse_dataset(dataset_paths):
    X = []
    y = []
    for index, dataset in enumerate(dataset_paths):
        print(f"[+] Parsing {dataset} data...")
        for fname in os.listdir(dataset):
            wav, sr = librosa.load(os.path.join(dataset, fname), sr=None)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
            X.append(D)
            y.append(index)
    return (X, y)

X, y = parse_dataset([BENJAMIN_DATA, JENS_DATA, JULIA_DATA, MARGARET_DATA, NELSON_DATA])

# 2. Divide data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

print("Longitud subconjunto de entrenamiento: ", len(X_train))
print("Longitud subconjunto de validación: ", len(X_val))
print("Longitud subconjunto de pruebas: ", len(X_test))

# 3. Build the model
def prep_dataset(X, y, shape):
    X_prep = np.array(X).reshape((len(X), shape))
    X_prep = X_prep.astype("float32") / 255
    y_prep = to_categorical(np.array(y))
    return (X_prep, y_prep)

X_train_prep, y_train_prep = prep_dataset(X_train, y_train, 1025*32)
X_val_prep, y_val_prep = prep_dataset(X_val, y_val, 1025*32)
X_test_prep, y_test_prep = prep_dataset(X_test, y_test, 1025*32)

# # Train the model with simple NN
# clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(10,), solver='sgd')
# clf.fit(X_train_prep, y_train)
#
# y_pred = clf.predict(X_val_prep)
#
# print(accuracy_score(y_val, y_pred))

# Train the model with a deep NN
network = models.Sequential()

network.add(layers.Dense(300, activation='relu', input_shape=(1025*32,)))
network.add(layers.Dense(200, activation='relu'))
network.add(layers.Dense(100, activation='relu'))
network.add(layers.Dense(5, activation='softmax'))

network.compile(loss='categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy', 'Precision'])

history = network.fit(X_train_prep,
                      y_train_prep,
                      epochs=30,
                      validation_data=(X_val_prep, y_val_prep))

pd.DataFrame(history.history).plot(figsize=(10, 7))
plt.grid(True)
plt.gca().set_ylim(0, 1.2)
plt.xlabel("epochs")
plt.show()

test_loss, test_acc, test_prec = network.evaluate(X_test_prep, y_test_prep)

print('test_acc:', test_acc)
print('test_prec:', test_prec)