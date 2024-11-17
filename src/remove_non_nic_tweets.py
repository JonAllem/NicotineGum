import pandas as pd
import os
import tqdm
import logging

log_folder = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_folder, exist_ok=True)

log_file = os.path.join(log_folder, 'remove_non_nic_tweets.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file, mode='w') 
stream_handler = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

if __name__ == '__main__':
    df = pd.DataFrame()
    logger.info('Reading files')
    orig_shape = 0
    for file in os.listdir('../data/clean/chunks'):
        temp = pd.read_parquet(f'../data/clean/chunks/{file}')
        logger.info(f'File {file} read with shape {temp.shape}')
        orig_shape += temp.shape[0]
        assert 'normalized_lemmatized_text' in temp.columns, 'normalized_lemmatized_text not in columns'
        temp = temp[(temp['normalized_lemmatized_text'].str.contains(r'\bnicotine\b') | temp['normalized_lemmatized_text'].str.contains(r'\bnic\b')) \
                    & temp['normalized_lemmatized_text'].str.contains(r'\bgum\b')]
        logger.info(f'File {file} filtered with shape {temp.shape}')
        df = pd.concat([df, temp])
        logger.info(f'File {file} concatenated with shape {df.shape}')
        del temp
    logger.info(f'Original size: {orig_shape}\t New size: {df.shape[0]}')
    logger.info(f'Saving file with shape {df.shape}')
    df.to_parquet('../data/clean/complete/nic_gum_tweets_2022.parquet',compression='gzip')
    if df.shape[0] < 10000: df.to_excel('../data/clean/complete/nic_gum_tweets_2022.xlsx', index=False)