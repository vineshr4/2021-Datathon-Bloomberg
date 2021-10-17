import pandas as pd
from inference_kmeans import *

def filter_by_yr(file_name, yr):
    df = pd.read_csv(file_name)
    pub_dates = df['publication_date']
    df_res = pd.DataFrame()

    for i in range(len(pub_dates)):
        pub_yr = pub_dates[i][0:4]
        if pub_yr == yr:
            row = df.iloc[i]
            df_res = df_res.append(row)

    return df_res

def make_dataframe():
    df = pd.read_csv (r'D:/TAMU/Events/Datathon2021/Bloomberg/2021-Datathon-Bloomberg/fed_doc_data_abs.csv')
    clustering = get_clustering()
    df['cluster'] = clustering
    return df

def get_rows_for_cluster(df, cluster):
    result_df = df.loc[df['cluster'] == cluster]
    return result_df

def get_topics_for_cluster_with_count(df, cluster):
    df.explode('topics')
    result_df = get_rows_for_cluster(df, cluster)
    result = result_df['topics'].value_counts()
    return result 

if __name__ == '__main__':
    filter_by_yr('fed_doc_data_abs.csv', '2020')
    df = make_dataframe()
    print(get_topics_for_cluster_with_count(df, 0))
