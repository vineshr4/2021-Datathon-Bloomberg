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


if __name__ == '__main__':
    data = pd.read_csv("fed_doc_data_abs.csv")
    x_values = np.array(encode_feats(data, 25, "abstract"))
    clustering = k_means(inference("topic_model_abstract.h5", x_values), 3)
    print(davies_bouldin_score(x_values, clustering))
