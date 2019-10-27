import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from get_spotify_data import normalize_features, get_ohe
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from numpy.random import seed
import random as rn 

seed(12345)
rn.seed(12345)

a_songs = pd.read_parquet("a_songs.parquet")
p_songs = pd.read_parquet("p_songs.parquet")

p_songs["LABEL"] = 1
a_songs["LABEL"] = 0

total_songs = pd.concat([p_songs, 
                         a_songs]).reset_index(drop=True)

numeric_features = ["popularity", "danceability", "energy",
                    "loudness", "speechiness", "acousticness",
                    "instrumentalness", "liveness", "valence", "tempo",
                    "duration_ms"] + ["timbre_{}".format(i) for i in range(12)] + ["pitches_{}".format(i) for i in range(12)]

categorical_features = ["key", "mode", "time_signature"]

X = pd.concat([normalize_features(total_songs, numeric_features), 
               get_ohe(total_songs, categorical_features), 
               total_songs[["id", "LABEL"]]], 1)
y = total_songs["LABEL"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=12345)

test_index = X_test[["id", "LABEL"]]
train_index = X_train[["id", "LABEL"]]

X_test = X_test.drop(columns=["id", "LABEL"])
X_train = X_train.drop(columns=["id", "LABEL"])

over_sampler = SMOTENC(categorical_features=list(range(35,len(X_train.columns))), random_state=12345)
X_train, y_train = over_sampler.fit_sample(X_train, y_train)

input = Input(shape=(X_train.shape[1],), name="Input")
x = Dense(5, activation='relu', name="Hidden_1")(input)
output = Dense(1, activation='sigmoid', name="Output")(x)
model = Model(inputs=input, outputs=output)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

adam = Adam(lr=0.005)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

history = model.fit(X_train, 
                    y_train,
                    shuffle=True,
                    batch_size=128,
                    epochs=100,
                    validation_data=[X_test.values, y_test.values],
                    callbacks=[early_stopping])

#pd.DataFrame(history.history).plot(title="Accuracy vs loss")

y_pred = model.predict(X_test.values)

print(classification_report(y_test, y_pred>.5))
print(confusion_matrix(y_test, y_pred>0.5))
test_scores = model.evaluate(X_test.values, y_test.values, verbose=2)

print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

model.save("spotipy_model.h5")
print("Done")
