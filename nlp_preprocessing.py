# Autres
import gensim
from string import punctuation
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import unicodedata
import sys
from unidecode import unidecode
from spacy_parser import read_spacy_from_arrays
# !python -m spacy download en_core_web_lg
# !python -m spacy download ru_core_news_lg 
#Importations des modèles spacy
import spacy
nlp_ru = spacy.load("ru_core_news_lg")
nlp_en = spacy.load("en_core_web_lg")
#Preprocess function
import string
from nltk.corpus import stopwords
#libraries datascience
import pandas as pd
import os
import numpy as np
# Other, utilities
from tqdm import tqdm
import datetime
import os
import pickle
from functools import partial  
import unicodedata  
from deep_translator import GoogleTranslator
from deep_translator.exceptions import NotValidPayload
from pandas.core.common import SettingWithCopyWarning
import warnings
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
# Fct pour obtenir des traductions rapides du Russe
def gtranslate(words, src='auto', lang='en') :
    res=[]
    for w in words :
        try :
            translated = GoogleTranslator(source=src, target=lang).translate(w)
            res.append(translated)
        except NotValidPayload :
            res.append(w)
    return(res)

# Fct pour normaliser les caractères spéciaux
normalize = partial(unicodedata.normalize, 'NFKD')  

# Fct pour Retirer les liens
def remove_html(text_data):
  txt = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', str(text_data))
  return txt
# Ftt pour retirer les @
def remove_ats(txt):
  txt = re.sub(r'@(\w+)', '', txt)
  return txt
def remove_hashtags(txt):
  txt = re.sub(r'#(\w+)', '', txt)
  return txt
  
# Fct pour Retirer les emojis
RE_EMOJI = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+"
)
def remove_emoji(text, repl=r' '):
    return RE_EMOJI.sub(repl, str(text))



tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))
def remove_unicode_punctuation(text):
    return text.translate(tbl)
def roberta_preprocess(text):
    new_text = []
    for t in text.split():
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', 'http', str(t))
        new_text.append(t)
    return " ".join(new_text)

from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA, LEMMA

names=['аляксандр','александр', 'лукашенка', 'лукашенко', 'ціханоўская', 'тихановская', 'тихановский', 'святлана', 'светлана', 'grodno', 'hrodna', 'gomel', 'homyel', 'колесникова', 'за']
def lemm(lang, text, use_spacy_data=False, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN', 'INTJ', 'NUM', 'X']):
    if lang=="english" :
        nlp=nlp_en
    elif lang=='russian' :
        nlp=nlp_ru
    doc = nlp(text)
    tokens = [(lambda x: x.text if x.text in names else x.lemma_)(token) for token in doc if token.pos_ in allowed_postags]
    
    return " ".join(tokens)


# Light pre-processing

all_punct_table=str.maketrans(string.punctuation, ' ' * len(string.punctuation))
whitespace_table=str.maketrans(string.whitespace, ' ' * len(string.whitespace))

def preprocess_light(doc, lower=True, normalize=True, punct_to_keep=None) :
    if punct_to_keep is not None :
        to_remove=' '.join([p for p in string.punctuation if p not in punct_to_keep])
        punct_table=str.maketrans(to_remove, ' ' * len(to_remove))
    else :
      punct_table = all_punct_table
# Normaliser l'unicode si besoin

    # On retire le html, les emojis, la ponctuation
    doc = remove_html(doc)
    doc = remove_emoji(doc)
    doc = remove_ats(doc)
    if normalize :
        doc=normalize(doc)
    # On retire la casse
    if lower :
        doc = doc.lower()
    # Supprime ponctuation
    doc = doc.translate(punct_table)

    # Supprime whitespaces
    doc = doc.translate(whitespace_table)
    doc = ' '.join([w for w in doc.split() if len(w) > 0])
    return(doc)

def batch_preprocess_light(docs,lower=True, normalize=True, punct_to_keep=None) :
    if punct_to_keep is not None :
        to_remove=' '.join([p for p in string.punctuation if p not in punct_to_keep])
        punct_table=str.maketrans(to_remove, ' ' * len(to_remove))
    else :
      punct_table = all_punct_table
# Normaliser l'unicode si besoin
    if normalize :
        docs = [normalize(doc) for doc in docs]
    # On retire la casse
    if lower :
        docs = [doc.lower() for doc in docs]
    # On retire le html, les emojis, la ponctuation, stopwords
    docs = [remove_html(doc) for doc in docs]
    docs = [remove_emoji(doc) for doc in docs]
    docs = [remove_ats(doc) for doc in docs]
    # Supprime ponctuation
    docs = [doc.translate(punct_table) for doc in docs]

    # Supprime whitespaces
    docs = [doc.translate(whitespace_table) for doc in docs]
    docs = [' '.join([w for w in doc.split() if len(w) > 0]) for doc in docs]
    return(docs)

# Complete pre-processing

def preprocess(docs, language, vocabulary_size=2000, use_spacy_data=False, make_ngrams=False, lemmatize=True, seuil_ngram = 25, min_count_ngram=10, max_df=0.95, min_df=3) :
  # On importe les stop_words de NLTK dans les deux langues, pas de pb à les supprimer deux fois ?
  # stop_words = stopwords.words(language)
  stop_words_ru = stopwords.words('russian')
  stop_words_ru.remove("за")
  stop_words_en = stopwords.words('english')
  stop_words = set(stop_words_en).union(set(stop_words_ru))
  if language=="russian" :
    connector_words=[]
  else :
    connector_words=gensim.models.phrases.ENGLISH_CONNECTOR_WORDS
  # On charge les documents
  preprocessed_docs_tmp = docs
  # On retire la casse
  preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
  # On retire le html, les emojis, la ponctuation, stopwords
  preprocessed_docs_tmp = [remove_html(doc) for doc in preprocessed_docs_tmp]
  preprocessed_docs_tmp = [remove_emoji(doc) for doc in preprocessed_docs_tmp]
  preprocessed_docs_tmp = [remove_ats(doc) for doc in preprocessed_docs_tmp]
  preprocessed_docs_tmp = [doc.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
  # Supprime whitespaces
  preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0]) for doc in preprocessed_docs_tmp]
  
  # On supprime les stop words, et aussi tout type de whitespace
  preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in stop_words]) for doc in preprocessed_docs_tmp]



  if lemmatize and not use_spacy_data :
    preprocessed_docs_tmp=[lemm(language, doc) for doc in tqdm(preprocessed_docs_tmp)]
    lemmatized_docs=preprocessed_docs_tmp
  else :
    lemmatized_docs=None
  removed={'before_ngrams':[], 'after_ngrams':[]}
  
  vectorizer = CountVectorizer(max_features=vocabulary_size, max_df=max_df, min_df=min_df)
  vectorizer.fit_transform(preprocessed_docs_tmp)
  # Ne pas utiliser "set(vectorizer.get_feature_names())" ??
  vocabulary = vectorizer.get_feature_names()
  preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in set(vocabulary)])
                                  for doc in preprocessed_docs_tmp]
  removed['before_ngrams']=vectorizer.stop_words_

  if make_ngrams : 
    bigram = gensim.models.Phrases([doc.split() for doc in preprocessed_docs_tmp], min_count=min_count_ngram, threshold=seuil_ngram, connector_words=connector_words, delimiter='_') # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[[doc.split() for doc in preprocessed_docs_tmp]], threshold=seuil_ngram, connector_words=connector_words, delimiter='_')  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    preprocessed_docs_tmp =[' '.join(trigram_mod[bigram_mod[doc.split()]]) for doc in preprocessed_docs_tmp]
    ngrams=trigram.export_phrases()
  
    # 2nd countvectorizer to re-compute vocab after making ngrams
    vectorizer = CountVectorizer(max_features=None, max_df=1.0, min_df=1)
    vectorizer.fit_transform(preprocessed_docs_tmp)
    vocabulary = vectorizer.get_feature_names()
  
    preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in set(vocabulary)])
                                    for doc in preprocessed_docs_tmp]

  vocab_out=list(vocabulary)
  removed['after_ngrams']=vectorizer.stop_words_
  
  preprocessed_docs, unpreprocessed_docs, deleted = [], [], []


  for i, doc in enumerate(preprocessed_docs_tmp):
    if len(doc) > 0:
      preprocessed_docs.append(doc)
      # Return a list of "low" preprocessed docs for BERT
      # roberta_preprocess is the preproc pipeline from roberta-xlm-twitter 
      unpreprocessed_docs.append(roberta_preprocess(docs[i]))
      # This preprocessing pipeline deletes documents that are empty after preprocessing.
      # We return a list of bool same size as original list to keep track of deleted docs
      deleted.append(0)
    else :
      deleted.append(1)
  if make_ngrams :
    return removed, lemmatized_docs, ngrams, preprocessed_docs, unpreprocessed_docs, vocab_out, deleted
  else :
    return removed, lemmatized_docs, None, preprocessed_docs, unpreprocessed_docs, vocab_out, deleted

def make_TM_dataset(infile, start, end, lang, save_path=None, vocabulary_size=2000, make_ngrams=False, max_df=0.7, min_df=2, seuil_ngram=8, min_count_ngram=2) :
  """Selects and pre-processeses a subset of data (dates, language), 
  with the option to save preprocessed data and OCTIS friendly datasets.
  Basic ouput : save_path/{start}_{end}_{lang}.csv
  Octis output : save_path/octis_preprocessed/{start}_{end}_{lang}/.csv'"""
  os.makedirs(save_path, exist_ok=True)

  if isinstance (start, datetime.datetime) :
    start_str = start.strftime('%Y-%m-%d')
  if isinstance (end, datetime.datetime) :
    end_str = end.strftime('%Y-%m-%d')
  data=pd.read_csv(infile)
  data=data.set_index(pd.to_datetime(data['created_at'], utc=True)).sort_index().loc[start_str:end_str]
  for c in [t for t in data.columns if 'id' in t] :
      data[c] = data[c].astype(str)
  data=data[['status_id', 'text', 'lang']]
  # for lang, df in [('russian', data[lambda x: x['lang']=='ru']), ('english', data[lambda x: x['lang']=='en'])] :
  for lang, df in [(lang, data[lambda x: x['lang']==lang[:2]])] :
      data=df.loc[start_str:end_str]
      tweets=data['text'].astype(str).tolist()
      # Preprocess selected
      removed_word_list, lemmatized, ngrams, preprocessed, unpreprocessed, vocab, deleted = preprocess(
          tweets, lang, vocabulary_size=vocabulary_size, make_ngrams=make_ngrams, max_df=max_df, 
          min_df=min_df, seuil_ngram=seuil_ngram, min_count_ngram=min_count_ngram
          )
      if save_path :
        # Output in a csv
        data.loc[:,"deleted"]=deleted
        data['deleted']=data['deleted'].astype(bool)
        data.loc[:,'preproc']=None
        data.loc[:,'unpreproc']=None
        data.loc[(~data.deleted), 'preproc']=preprocessed
        data.loc[(~data.deleted), 'unpreproc']=unpreprocessed
        if isinstance (start, datetime.datetime) :
            start_str = start.strftime('%Y%m%d')
        else :
            start="".join(start.split("-"))
        if isinstance (end, datetime.datetime) :
            end_str = end.strftime('%Y%m%d')
        else :
            end="".join(end.split("-"))
        data.set_index('status_id').to_csv(save_path+f'/{start_str}_{end_str}_{lang}.csv')
        # Output vocab/data in a separate folder to use with OCTIS models
        # Write the preprocessed dataset
        folder=save_path+f'/octis_preprocessed/{start_str}_{end_str}_{lang}'
        os.makedirs(folder, exist_ok=True)
        # Write the Vocabulary
        try:
            os.remove(folder+'/vocabulary.txt')
        except OSError:
            pass
        with open(folder+'/vocabulary.txt', 'a') as f:
            for k,w in enumerate(vocab) :
                if k < len(vocab) :
                    f.write(w+"\n")
                else : 
                    f.write(w)
        # Write the corpus
        pd.Series(preprocessed).to_csv(folder +'/corpus.tsv', sep='\t', header=False, index=False)
        # Write the unpreprocessed dataset
        folder=save_path+f'/octis_unpreprocessed/{start_str}_{end_str}_{lang}'
        os.makedirs(folder, exist_ok=True)
        pd.Series(unpreprocessed).to_csv(folder +'/corpus.tsv', sep='\t', header=False, index=False)

      return preprocessed, unpreprocessed, vocab, deleted