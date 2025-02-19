import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def generate_data(seq_length=50, num_samples=1000):
    x = np.linspace(0, 100, num_samples)
    data = np.sin(x)
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == '__main__':
    seq_length = 50
    X, y = generate_data(seq_length=seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = build_model((seq_length, 1))
    model.fit(X, y, epochs=10, batch_size=32)
    prediction = model.predict(X[-1].reshape(1, seq_length, 1))
    print("Next predicted value:", prediction[0][0])
