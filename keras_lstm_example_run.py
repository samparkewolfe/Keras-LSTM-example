from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import keras
import numpy as np
import h5py

model = keras.models.load_model("model.h5")

input = np.random.rand(1, 1024)
output = model.predict(input)

print(output)