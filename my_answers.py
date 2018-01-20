import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.regularizers import L1L2
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    for i in range(len(series)-window_size):
        # From i to i+window_size (excluded) input characters
        X.append(series[i:i+window_size])
        # result in i+window_size output character
        y.append(series[i+window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = [' ', '!', ',', '.', ':', ';', '?']
    charset = ''.join(punctuation) + 'abcdefghijklmnopqrstuvwxyz'
    # keep only valid characters, in one run we should clean the text
    return ''.join([c for c in text if c in charset])


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    for i in range(0, len(text) - window_size, step_size):
        # from i to i+windows+size (excluded)
        inputs.append(text[i:i+window_size])
        # only the character at index i+window_size
        outputs.append(text[i+window_size])
    return inputs, outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    # I've seen only worse perfornmances after adding some regularization and dropout,
    # but I hadn't had the time to explore the issue deeper
    model.add(LSTM(200, input_shape=(window_size, num_chars), dropout=0.0))
    # adding dropout and/or regularizer made the overall loss higher
    # 3, kernel_regularizer=L1L2(l1=0.1, l2=0.1)
    model.add(Dense(num_chars))
    model.add(Activation("softmax"))
    return model
