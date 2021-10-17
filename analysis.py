import pandas as pd
from inference_kmeans import *
import matplotlib.pyplot as plt
import ast


def filter_by_year(file_name, year):
    df = pd.read_csv(file_name)
    pub_dates = df['publication_date']
    df_res = pd.DataFrame()

    for i in range(len(pub_dates)):
        pub_yr = pub_dates[i][0:4]
        if pub_yr == year:
            row = df.iloc[i]
            df_res = df_res.append(row)

    return df_res


def topics_by_clustering(df, clustering):
    clustered_topics = [[] for t in range(max(clustering) + 1)]

    topics = df["topics"].tolist()
    topics = [ast.literal_eval(i) for i in topics]

    for i in range(0, len(clustering)):
        curr_topics = topics[i]
        clustered_topics[clustering[i]] += curr_topics

    return clustered_topics


def topic_frequency(clustered_topics):
    all_topics = []
    all_frequency = []
    num_clusters = len(clustered_topics)
    for i in range(0, num_clusters):
        curr_topics = clustered_topics[i]
        topics = []
        frequencies = []
        for j in range(0, len(curr_topics)):
            if curr_topics[j] not in topics:
                topics.append(curr_topics[j])
                frequencies.append(curr_topics.count(curr_topics[j]))

        s = [x for _, x in sorted(zip(frequencies, topics), reverse=True)]
        frequencies.sort(reverse=True)

        all_topics.append(s)
        all_frequency.append(frequencies)

    return all_topics, all_frequency


if __name__ == '__main__':
    data = pd.read_csv("fed_doc_data_abs.csv")
    # x_values = np.array(encode_feats(data, 25, "abstract"))
    x_values = np.load("x_values.npy")
    cluster = k_means(inference("topic_model_abstract.h5", x_values), 3)
    topic_clusters = topics_by_clustering(data, cluster)
    m_topics, m_frequencies = topic_frequency(topic_clusters)

    for i in range(0, len(m_topics)):
        print(m_topics[i])
        print(m_frequencies[i])
