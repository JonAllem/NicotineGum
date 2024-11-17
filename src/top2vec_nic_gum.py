#%%
import pandas as pd
from tqdm import tqdm
if __name__ == '__main__':
    data=pd.read_parquet('../data/precollected/partial_cleaned_data_unique.parquet')
    tqdm.pandas()
    data.normalized_lemmatized_text = data.normalized_lemmatized_text.progress_apply(lambda x: x.replace('@person','').strip())
    data.head()
    