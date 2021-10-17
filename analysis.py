import pandas as pd


def filter_by_yr(file_name, yr):
    df = pd.read_csv(file_name)
    pub_dates = df['publication_date']
    df_res = pd.DataFrame()

    for i in range(len(pub_dates)):
        pub_yr = pub_dates[i][0:4]
        if pub_yr == yr:
            row = df.iloc[i]
            df_res.append(row)

    return df_res


if __name__ == '__main__':
    filter_by_yr('fed_doc_data_abs.csv', '2020')
