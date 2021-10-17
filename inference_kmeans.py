import pandas as pd

from topic_prediction import *
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from analysis import *


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
        cluster_dict[i] = pd.DataFrame()

    # populate cluster dict
    for i in range(len(clusters)):
        cluster_val = int(clusters[i])
        row = df.iloc[i]
        cluster_dict[cluster_val] = cluster_dict[cluster_val].append(row)

    return cluster_dict


def common_regulations(yr_range):
    # get all cluster dicts within yr range
    relevant_cluster_dicts = []

    for yr in range(yr_range[0], yr_range[1] + 1):
        yr_frames = filter_by_yr('fed_doc_data_abs.csv', str(yr))
        x_values = np.array(encode_feats(yr_frames, 25, "abstract"))
        clustering = k_means(inference("topic_model_abstract.h5", x_values), 3)

        cluster_dict = get_cluster_dict(yr_frames, clustering, 3)
        relevant_cluster_dicts.append(cluster_dict)

    return relevant_cluster_dicts


if __name__ == '__main__':
    # data = pd.read_csv("fed_doc_data_abs.csv")
    # x_values = np.array(encode_feats(data, 25, "abstract"))
    # clustering = k_means(inference("topic_model_abstract.h5", x_values), 3)
    # print(davies_bouldin_score(x_values, clustering))
    # print(clustering)
    #
    # get_cluster_dict(data, clustering, 3)
    common_regulations((2001, 2006))

