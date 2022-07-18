# NLP libraries
# from transformers import AutoModelForSequenceClassification
# from transformers import BertTokenizerFast
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import datasets as dts
import unicodedata
from nlp_preprocessing import remove_html, remove_hashtags, remove_ats, remove_emoji, remove_unicode_punctuation, roberta_preprocess
#libraries datascience
import pandas as pd
import numpy as np
# Other, utilities
from tqdm import tqdm
import warnings
import sys
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

## Nettoyage et tokenization

tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))
def remove_unicode_punctuation(text):
    return text.translate(tbl)

modelnames=['blanchefort/rubert-base-cased-sentiment-rusentiment', "cardiffnlp/twitter-roberta-base-sentiment-latest"]

def sentiments(texts, lang, batch_size=1):
    texts=[roberta_preprocess(text) for text in texts]
    res=[]
    if lang=='english' :
        m_name=modelnames[1] 
    else :
        m_name=modelnames[0] 
    pipe = pipeline(task="text-classification", model=m_name, device=0)
    for r in tqdm(pipe(texts, batch_size=batch_size)) :
        res.append(r)
    return res
def series_sentiments(series, lang, batch_size=1):
    texts=series.apply(lambda x: roberta_preprocess(x))

    dataset = dts.Dataset.from_pandas(texts)
    res=[]
    if lang=='english' :
        m_name=modelnames[1] 
    else :
        m_name=modelnames[0] 
    pipe = pipeline(task="text-classification", model=m_name, device=0)
    for out in tqdm(pipe(KeyDataset(dataset, "text"),batch_size=batch_size)) :
    # results=pipe(texts)
        res.append(out)
    return res

def sentiments_from_df(df, lang, batch_size=1):
    dataset = dts.Dataset.from_pandas(df)
    df['texts']=df.text.apply(lambda x:roberta_preprocess(x))
    res=[]
    if lang=='english' :
        m_name=modelnames[1] 
    else :
        m_name=modelnames[0] 
    pipe = pipeline(task="text-classification", model=m_name, device=0)
    for out in tqdm(pipe(KeyDataset(dataset, "text"), batch_size=batch_size)) :
    # results=pipe(texts)
        res.append(out)
    return res



def compute_sentiment(infile="data/beltweets_nlp.csv", outfile = "data/beltweets_nlp_sentiment.csv") :
# Importation des datasets
    data = pd.read_csv(infile, lineterminator='\n')
    data['created_at']=pd.to_datetime(data['created_at'], utc=True)
    for c in [t for t in data.columns if 'id' in t] :
        data[c] = data[c].astype(str)
    # Application fonction
    for lang in  ['english', 'russian'] :
        data.loc[(data.lang == lang[:2]), 'sentiment'] = sentiments(data.loc[(data.lang == lang[:2])].text.to_list(), lang, batch_size=16)
    data.to_csv(outfile)