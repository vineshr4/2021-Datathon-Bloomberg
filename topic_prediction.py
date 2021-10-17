import tensorflow as tf
import numpy as np
import pandas as pd
import enchant
import math


def get_all_topics(df):
    all_topics = []
    topics_mat = df["topics"].tolist()
    for i in range(0, len(topics_mat)):
        all_topics += topics_mat[i]

    return list(set(all_topics))


def encode_labels(df, all_topics):
    t = df["topics"].tolist()
    v = []
    for topics in t:
        encoded = [0] * len(all_topics)
        for i in range(0, len(topics)):
            if topics[i] in all_topics:
                encoded[all_topics.index(topics[i])] = 1

        v.append(encoded)

    return v


def occurrence_sort(all_words):
    d = {}
    word = enchant.Dict("en_US")
    for i in range(0, len(all_words)):
        if all_words[i].lower() not in d.keys() and word.check(all_words[i]):
            d[all_words[i].lower()] = all_words.count(all_words[i])

    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True))


def get_all_words(df, text_col):
    all_words = []
    titles = df[text_col].tolist()

    for i in titles:
        try:
            all_words += i.split(" ")

        except AttributeError:
            all_words += []

    return all_words


def encode_feats(df, max_words, text_col):
    values = []

    all_words = get_all_words(df, text_col)
    word_freqs = occurrence_sort(all_words)

    word_freqs = list(word_freqs.keys())[0:max_words]
    word_freqs = [t.lower() for t in word_freqs]

    text = df[text_col].tolist()

    for i in range(0, len(text)):
        if str(text[i]) == "nan":
            text[i] = ""

        text[i] = text[i].split(" ")
        text[i] = [i.lower() for i in text[i]]
        s = [0] * max_words
        for j in range(0, len(text[i])):
            if text[i][j] in word_freqs:
                s[word_freqs.index(text[i][j])] += 1

        values.append(s)

    return values


if __name__ == "__main__":
    data = pd.read_csv("fed_doc_data_abs.csv")
    x_values = np.array(encode_feats(data, 15, "title"))
    y_values = np.array(encode_labels(data, get_all_topics(data)))

    samples = x_values.shape[1]
    dims = x_values.shape[1]
    output_length = y_values.shape[1]

    print("Output length:", output_length)
    print(x_values)
    print(y_values)

    inputs = tf.keras.Input(shape=dims, )
    l1 = tf.keras.layers.Dense(20)(inputs)
    l2 = tf.keras.layers.Dense(30)(l1)
    l3 = tf.keras.layers.Dense(output_length)(l2)
    output = tf.keras.layers.Dense(output_length)(l3)

    model = tf.keras.models.Model(inputs, output)

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])

    print(model.summary())

    model.fit(x_values, y_values, batch_size=200, epochs=100)

    model.save("topic_model_title.h5")
