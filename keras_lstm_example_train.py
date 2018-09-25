from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import h5py

max_features = 1024

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

x_train = np.random.rand(1, max_features)
y_train = np.random.rand(1, 1)

x_test = np.random.rand(1, max_features)
y_test = np.random.rand(1, 1)

model.fit(x_train, y_train, batch_size=16, epochs=1)
score = model.evaluate(x_test, y_test, batch_size=16)

model.save("model.h5")