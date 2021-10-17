import pandas as pd

from topic_prediction import *
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score


def inference(model_path, x):
    loaded_model = tf.keras.models.load_model(model_path)
    layer_name = "dense"
    layer_output = loaded_model.get_layer(layer_name).output
    new_model = tf.keras.models.Model(loaded_model.input, outputs=layer_output)

    return new_model.predict(x)


def k_means(train_data, k):
    km = KMeans(n_clusters=k, random_state=0).fit(train_data)

    return km.labels_

def get_clustering():
    data = pd.read_csv("fed_doc_data_abs.csv")
    x_values = np.array(encode_feats(data, 25, "abstract"))
    clustering = k_means(inference("topic_model_abstract.h5", x_values), 3)
    return clustering


def get_cluster_dict(df, clusters, num_clusters):
    # create cluster dict
    cluster_dict = {}
    for i in range(num_clusters):
        cluster_dict[i] = []

    # populate cluster dict
    for i in range(len(clusters)):
        cluster_val = int(clusters[i])
        row = df.iloc[i]
        cluster_dict[cluster_val].append(row)

    return cluster_dict


if __name__ == '__main__':
    data = pd.read_csv("fed_doc_data_abs.csv")
    # x_values = np.array(encode_feats(data, 25, "abstract"))
    # clustering = k_means(inference("topic_model_abstract.h5", x_values), 3)
    # # write clustering to a file
    # out = open('clusters.txt', 'w')
    # clusters_str = ''
    # for cluster in clustering:
    #     clusters_str += str(cluster) + ','
    # out.write(clusters_str[:-1])
    # print(davies_bouldin_score(x_values, clustering))
    # print(clustering)
    clustering = open('clusters.txt', 'r').readlines()[0].split(',')
    print(clustering)

    get_cluster_dict(data, clustering, 3)
