# LSTM for sequence classification
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility

if __name__ == '__main__':
    max_review_length = 100
    embedding_length = 32
    num_titles = 5000

    model = Sequential()
    model.add(Embedding(num_titles, embedding_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(10))
    model.compile(loss='crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # model.fit(X_train, y_train, epochs=3, batch_size=64)
    print('hello')
