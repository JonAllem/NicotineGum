#! python {workspaceFolder}/src/get_raw_data.py
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.sql import select
import time
import os
import logging
from argparse import ArgumentParser, ArgumentTypeError
import json
from tqdm import tqdm

log_folder = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_folder, exist_ok=True)

log_file = os.path.join(log_folder, 'collect_tweets.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file, mode='w') 
stream_handler = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def read_auth(path):
    with open(path) as auth_f:
        auth = json.load(auth_f)
    auth_f.close()
    return auth

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser.add_argument('-ys','--years', nargs='+', type=int, help='Calendar years for input', required=True)
    # parser.add_argument('-irt','--include_retweets', default=False, type=str2bool, required=True)
    # args = parser.parse_args()
    auth = read_auth('../auth/auth.json')
    try:
        logger.info("Starting script...")
        engine = create_engine(f"mysql+mysqlconnector://{auth['user']}:{auth['pass']}@{auth['host']}/{auth['database']}")
        conn = engine.connect()
        logger.info("Connected to database.")
        
        date_ranges = pd.date_range(start='2021-01-01', end='2023-01-01', freq='M')  # adjust the frequency as needed
        chunks = []
        total_t = time.time()
        for start_date, end_date in tqdm(zip(date_ranges, date_ranges[1:]), total=len(date_ranges)-1):
            t = time.time()
            query = select('*').select_from(
                text('tweets')).where(
                    text(f"isRetweet = 0 AND createdAt >= '{start_date.strftime('%Y-%m-%d')}' AND createdAt < '{end_date.strftime('%Y-%m-%d')}'")
                )
            chunk = pd.read_sql_query(query, conn)
            logger.info(f"Query for data range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} executed and data loaded into DataFrame in {time.time()-t}.")
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Query for year 2021 & 2022 executed and data loaded into DataFrame in {time.time()-total_t}.")
        
        df.to_feather(f"../data/raw/nic_gum_tweets_no_retweets_2021_2022.feather", compression='zstd')
        logger.info("Data saved to feather file with zstd compression.")
    
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        conn.close()
        logger.info("Database connection closed.")
