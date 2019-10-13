from pathlib import Path
import time
import datetime
import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix
from io import StringIO
import itertools

""" GENERAL TOOL FUNCTIONS """
cache_path = lambda filename: Path(f"""cache/{filename}""")
figures_path = lambda filename: Path(f"""figures/{filename}""")
tables_path = lambda filename: Path(f"""tables/{filename}""")
output_path = lambda filename: Path(f"""output/{filename}""")

def create_directories(date):
    """creates sub-directories for monthly data, if they don't exist already"""
    Path(f"""cache/{date}""").mkdir(exist_ok=True, parents=True)
    Path(f"""figures/{date}""").mkdir(exist_ok=True, parents=True)
    Path(f"""tables/{date}""").mkdir(exist_ok=True, parents=True)
    Path(f"""output/{date}""").mkdir(exist_ok=True, parents=True)

def get_dates(last = '2019_04'):
    years = [2015, 2016, 2017, 2018, 2019]
    months = list(range(1,13))
    dates = [f"{y}_{m:02d}" for y, m in itertools.product(years, months)]

    return dates[:dates.index(last)+1]

def load_defaults():
    return pd.read_csv(cache_path('defaults.csv'), squeeze=True, header=None,
                        dtype=str)

def CSR(df, row, col, data):
    """ Returns a csr matrix representation of a dataframe """
    row = df[row]
    col = df[col]
    data = df[data]

    return csr_matrix((data, (row, col)))

def get_log(df):
    """Returns log values of a dataframe
        - replaces log(0) -inf values with 0
    """
    return df.where(lambda df: df > 0).pipe(np.log).fillna(0)

def get_upper_edges(adj):
    """
    Converts matrix to list of undirected, loopless edges
    Takes a square matrix
    Returns a list of edges from upper right corner
    Removes selfloops
    """
    copy = adj.copy()

    keep = (np.triu(np.ones(copy.shape), k=1).astype('bool')
    .reshape(copy.size))
    
    return copy.stack()[keep].sort_values(ascending=False)

def standardise(s):
    return (s - s.mean())/s.std()

def normalise(s):
    return (s-s.min())/(s.max()-s.min())


""" GCS TOOL FUNCTIONS """

def dataset_id(date):
    return date + '_final'

# BIGQUERY
from google.cloud import storage
from google.cloud import bigquery
from io import StringIO
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

bigquery_client = lambda: bigquery.Client.from_service_account_json(Path('credentials/gcs_service_account.json'))

def create_dataset(dataset):
    bigquery_client().create_dataset(dataset, exists_ok=True)

def delete_table(dataset_id, table_name):
    from google.cloud.exceptions import NotFound
    
    dataset = bigquery_client().dataset(dataset_id)
    table = dataset.table(table_name)

    try:
        bigquery_client().delete_table(table)
    except NotFound:
        return
    
def run_job(query, dataset_id=None, table_id=None):
    client = bigquery_client()

    job_config = bigquery.QueryJobConfig()
    job_config.use_legacy_sql = False

    if dataset_id and table_id:
        table_ref = client.dataset(dataset_id).table(table_id)
        job_config.destination = table_ref

    query_job = client.query(query, job_config=job_config)

    return query_job.result()

def extract_table(dataset_id, table_id, bucket_name, blob_name):
    print(f"Extracting {dataset_id}.{table_id} to {bucket_name}/{blob_name}")

    client = bigquery_client()
    destination_uri = 'gs://{}/{}'.format(bucket_name, blob_name)
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    extract_job = client.extract_table(
        table_ref,
        destination_uri,
        location='US')
    extract_job.result()


# Storage buckets
storage_client = lambda: storage.Client.from_service_account_json(Path('credentials/gcs_service_account.json'))

def create_bucket(dataset):
    storage_client().create_bucket(dataset)

def delete_blob(date, bucket_name, blob_name):
    from google.cloud.exceptions import NotFound
    
    bucket = storage_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)

    try:
        blob.delete()

    except NotFound:
        return

def blob_to_df(blob):
    string = blob.download_as_string()
    data = StringIO(string.decode("utf-8"))

    return pd.read_csv(data, index_col=0)

def get_blob(bucket_name, blob_name, df=False):
    """df is True if want to return blob as pandas df"""
    
    print(f"Getting {blob_name} from {bucket_name}")
    client = storage_client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if df:
        return blob_to_df(blob)
    else:
        return blob

def insert_to_name(orig, add, ext='.csv'):
    index = orig.find(ext)

    return orig[:index] + add + orig[index:]

def store_blob(df, bucket_name, blob_name):
    client = storage_client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(df.to_csv())

def clear_date(date):
    from google.cloud.exceptions import NotFound
    try:
        bigquery_client().delete_dataset(date, delete_contents=True,
                                         not_found_ok=True)
    except NotFound:
            print()
    try:
        storage_client().get_bucket(f"emg-{date}").delete()
    except NotFound:
            print()
    


# LOAD ANCILLARY DATA

def sub_topics():
    return (pd.read_csv(output_path("subreddit_labels.csv"),
            index_col=0, header=0)['topic'].str.split(', ', expand=True)[0])

def write_sub_topics(date):
    topics = pd.DataFrame({'topic':pd.read_csv(output_path("subreddit_labels.csv"),
        index_col=0, header=0)['topic'].str.split(', ', expand=True)[0]})

    bucket_name = dataset_id(date)
    store_blob(df=topics, bucket_name=bucket_name, blob_name='subreddit_labels.csv')

"""
def write_bots(date):
    botlist = pd.DataFrame({'author':('AutoModerator', '[deleted]', 'autotldr', 'Mentioned_Videos',
                            'TweetPoster', 'xkcd_transcriber', 'imgurtranscriber',
                            'The-Paranoid-Android', 'TotesMessenger','HelperBot_',
                            'Bot_Metric', 'B0tRank', 'KeepingDankMemesDank', 'Marketron-I',
                            'MTGCardFetcher',
                            )})
    botlist.to_csv(cache_path("confirmed_bots.csv"), header=False) 
    
    bucket_name = dataset_id(date)
    store_blob(df=botlist, bucket_name=bucket_name, blob_name='botlist.csv')

    
def load_bots():
    return pd.read_csv(cache_path("confirmed_bots.csv"), index_col=0)
"""




