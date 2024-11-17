from lingua import Language, LanguageDetectorBuilder
from pandarallel import pandarallel
import nltk
from nltk.util import ngrams
import pandas as pd
import numpy as np
import nltk
import re
from argparse import ArgumentParser, ArgumentTypeError
from tqdm import tqdm
from time import time
import os
import logging

log_folder = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_folder, exist_ok=True)

log_file = os.path.join(log_folder, 'clean_tweets.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file, mode='w') 
stream_handler = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class DataCleaner:
    """
    Cleans the twitter database. Performs the following - 
    1. Removes retweets (already done with SQL query filter)
    2. Normalize tweets
    3. Remove non - english tweets
    """
    def __init__(self, logger:logging.Logger, multiprocess:bool) -> None:
        self.url_regex = re.compile(
            r'^(?:http|ftp)s?://' # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
            r'localhost|' #localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
            r'(?::\d+)?' # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        self.logger = logger
        self.multiprocess = multiprocess
        self.logger.info('Initializing Data Cleaner...')
        self.lemmaztizer = nltk.WordNetLemmatizer()
        pandarallel.initialize(progress_bar=True,verbose=0,nb_workers=26)
    
    
    def parallel_english_detection_copy(self,text):
        from lingua import LanguageDetectorBuilder
        from re import sub
        detector = LanguageDetectorBuilder.from_all_languages_with_latin_script().build()
        return detector.compute_language_confidence_values(sub("(?P<url>https?://[^\s]+)","",text).strip())[:5]

    def normalize_tweet(self,text:str):
        """
        """
        words, lemma, hashtags, tagged_ids = [], [], set(), []
        from nltk.tokenize import TweetTokenizer
        import nltk
        import string
        STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
        self.tokenizer = TweetTokenizer()
        import numpy as np
        import re 
        def _wordnet_pos_code(tag):
            """Converts nltk stanford tags to wordnet pos tags"""
            if tag.startswith('NN'):
                return nltk.corpus.wordnet.NOUN
            elif tag.startswith('VB'):
                return nltk.corpus.wordnet.VERB
            elif tag.startswith('JJ'):
                return nltk.corpus.wordnet.ADJ
            elif tag.startswith('RB'):
                return nltk.corpus.wordnet.ADV
            else:
                return nltk.corpus.wordnet.NOUN
        
        import psutil
    
        if psutil.virtual_memory()[2]>=95:
            print('Usable Memory Crossed 95%\n', psutil.virtual_memory())
            raise MemoryError('Usable Memory Crossed 95%')
            
        for token, tag in nltk.pos_tag(self.tokenizer.tokenize(text.lower())):
            if token in STOP_WORDS: continue
            if len(token)<2 or self.url_regex.search(token): continue
            elif all(c in string.punctuation for c in token): continue
            elif token[0]=='@': 
                tagged_ids.append(token)
                token, tag = '@person', 'NNP' # replace a mention with a generic 'person' mention, and tag as a proper noun (NNP)
            elif token[0]=='#': hashtags.add(token)
            words.append(token)
            lemma.append(self.lemmaztizer.lemmatize(token, pos = _wordnet_pos_code(tag)))
        

        return ' '.join(lemma), list(hashtags) if len(hashtags)==0 else [], tagged_ids, list(re.findall("(?P<url>https?://[^\s]+)", text))
        
    def remove_non_english_tweets(self,df:pd.DataFrame):
        
        def parallel_english_detection(text):
            import psutil
            if psutil.virtual_memory()[2]>=95:
                print('Usable Memory Crossed 95%\n', psutil.virtual_memory())
                self.logger.error(MemoryError('Usable Memory Crossed 95% while performing parallel english detection'))
                raise MemoryError('Usable Memory Crossed 95% while performing parallel english detection')
            from lingua import LanguageDetectorBuilder
            from lingua import Language
            from re import sub
            ENGLISH = Language.ENGLISH
            detector = LanguageDetectorBuilder.from_all_languages_with_latin_script().build()
            for lang, _ in detector.compute_language_confidence_values(sub("(?P<url>https?://[^\s]+)","",text).strip())[:5]:
                if lang==ENGLISH: return True
            return False

        self.logger.info('Removing non english tweets')
        pd.options.mode.chained_assignment = None
        t1 = time()
        if self.multiprocess:
            df=df[df.text.parallel_apply(parallel_english_detection)]
        else:
            from lingua import LanguageDetectorBuilder
            from lingua import Language
            from re import sub
            ENGLISH = Language.ENGLISH
            detector = LanguageDetectorBuilder.from_all_languages_with_latin_script().build()
            tqdm.pandas(desc='Removing Non English Tweets')
            df=df[df.text.progress_apply(
                lambda x: any(
                lang==ENGLISH for lang, _ in detector.compute_language_confidence_values(sub("(?P<url>https?://[^\s]+)","",x).strip())[:5])
                )]
        self.logger.info('Removed Non English tweets in '+str(time()-t1)+' seconds. Data Size revised to...\t', len(df))
        pd.options.mode.chained_assignment = 'warn'
        return df
    
    def normalize(self,df:pd.DataFrame) -> pd.DataFrame:
        self.logger.info('Normalizing tweets, extracting hash tags, text links, and tagged profiles...')
        if self.multiprocess:
            df[['normalized_lemmatized_text','hash_tags','tagged_profiles','text_links']] = pd.DataFrame(df['text'].parallel_apply(self.normalize_tweet).tolist(),index = df.index)
        else:
            tqdm.pandas(desc='Normalizing Tweets')
            df[['normalized_lemmatized_text','hash_tags','tagged_profiles','text_links']] = pd.DataFrame(df['text'].progress_apply(self.normalize_tweet).tolist(),index = df.index)

        return df

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def process_file_uncombined(file:str, data_path:str, clean_path:str, logger:logging.Logger, args):
    if file.endswith('.feather'):
                logger.info('Cleaning '+file)
                cleaner = DataCleaner(logger=logger, multiprocess=args.multiprocessing)
                
                df = pd.read_feather(os.path.join(data_path,file))
                logger.info('Read '+str(len(df))+' tweets')
                try:
                    df = cleaner.remove_non_english_tweets(df)
                    
                    df = cleaner.normalize(df)
                    logger.info('Normalized tweets, extracted hash tags, text links, and tagged profiles')
                    df.to_parquet(os.path.join(clean_path,'clean_'+os.path.splitext(file)[0]+'.parquet'))
                    logger.info('Saved cleaned tweets Chunk to '+os.path.join(clean_path,'clean_'+file))
                except Exception as e:
                    logger.error('Error while cleaning '+file+'\n'+str(e))
                    raise Exception('Error while cleaning '+file+'\n'+str(e))
                finally:
                    del df
                    logger.info('Cleaned '+file)
    else:
        raise ValueError('File '+file+' is not a feather file')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-co','--combine_output', default=True, type=str2bool)
    parser.add_argument('-m', '--multiprocessing', default=True, type=str2bool)
    parser.add_argument('-bm', '--batch_multiprocessing', default=True, type=str2bool)
    args = parser.parse_args()

    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

    logger.info('Starting Data Cleaning...')
    logger.info('Multiprocessing: '+str(args.multiprocessing))
    logger.info('Combine Output: '+str(args.combine_output))
    data_path = '../data/raw/chunks' if not args.combine_output else '../data/raw/complete/nic_gum_tweets_no_retweets_2022.feather'
    clean_path = '../data/clean/chunks' if not args.combine_output else '../data/clean/complete/clean_nic_gum_tweets_no_retweets_2022.parqut'
    
    if args.combine_output:
        logger.info('Cleaning '+data_path)
        cleaner = DataCleaner(logger=logger, multiprocess=args.multiprocessing)
        df = pd.read_feather(data_path)
        logger.info('Read '+str(len(df))+' tweets')
        try:
            df = cleaner.remove_non_english_tweets(df)
            df = cleaner.normalize(df)
            logger.info('Normalized tweets, extracted hash tags, text links, and tagged profiles')
            df.to_parquet(clean_path)
            logger.info('Saved cleaned tweets Chunk to '+clean_path)
        except Exception as e:
            logger.error('Error while cleaning '+data_path+'\n'+str(e))
        finally:
            del df
            logger.info('Cleaned')
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        if args.batch_multiprocessing:
                num_workers = multiprocessing.cpu_count() - 3
                num_workers = max(1, num_workers)

                files = [f for f in os.listdir(data_path)]

                with ProcessPoolExecutor(max_workers=num_workers) as executor:  
                    executor.map(process_file_uncombined, files)
        else:
            for file in os.listdir(data_path):                
                process_file_uncombined(file, data_path, clean_path, logger, args)