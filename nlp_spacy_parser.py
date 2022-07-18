#autres librairies nlp (pre processing...)
import nltk
import gensim
from string import punctuation
from bs4 import BeautifulSoup
import re
import shutil
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# # 1st time we use the notebook
# !python -m spacy download en_core_web_lg
# !python -m spacy download ru_core_news_lg 
#Importations des modèles spacy
import spacy
nlp_ru = spacy.load("ru_core_news_lg")
nlp_en = spacy.load("en_core_web_lg")
#Preprocess function
import string
import warnings
from nltk.corpus import stopwords
#libraries datascience
import pandas as pd
import os
import numpy as np
# Other, utilities
import pickle
from tqdm import tqdm
import os
import time
import pickle
import json
import requests

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

# Importation des datasets

#orig
# data = pd.read_csv("datasets/beltweets_nlp.csv", lineterminator='\n')
data = pd.read_csv("datasets/beltweets_nlp.csv")

data['created_at']=pd.to_datetime(data['created_at'], utc=True)
for c in [t for t in data.columns if 'id' in t] :
    data[c] = data[c].astype(str)



## Nettoyage et tokenization

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
def remove_emoji(text):
    return RE_EMOJI.sub(r'', str(text))

import unicodedata
import sys

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



dir(spacy.attrs)

from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA, LEMMA, LANG, ENT_ID, ENT_IOB, TAG



names=['аляксандр','александр', 'лукашенка', 'лукашенко', 'ціханоўская', 'тихановская', 'тихановский', 'святлана', 'светлана', 'grodno', 'hrodna', 'gomel', 'homyel', 'колесникова', 'за']
def lemm(lang, text, use_ner=False, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN', 'INTJ', 'NUM', 'X']):
    if lang=="english" :
        nlp=nlp_en
    elif lang=='russian' :
        nlp=nlp_ru
    doc = nlp(text)
    tokens = [(lambda x: x.text if x.text in names else x.lemma_)(token) for token in doc if token.pos_ in allowed_postags]
    
    return " ".join(tokens)

# Specific named entities which we want to keep as is
names=['аляксандр','александр', 'лукашенка', 'лукашенко', 'ціханоўская', 'тихановская', 'тихановский', 'святлана', 'светлана', 'grodno', 'hrodna', 'gomel', 'homyel', 'колесникова', 'за']
def lemm_ru(text, use_ner=False, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN', 'INTJ', 'NUM', 'X']):
    nlp = nlp_ru
    doc = nlp(text)
    # Si on veut traiter à part le lemme des entités nommées
    # namelist = [(lambda x: x.text if x.text in ['лукашенко', 'лукашенка'] else x.lemma_) (name) for name in doc.ents]
    # On crée une liste de tokens à l'exclusion de ces dernières
    # tokens = [token.lemma_ for token in doc if token.ent_type==0 and token.pos_ in allowed_postags]
    tokens = [(lambda x: x.text if x.text in names else x.lemma_)(token) for token in doc if token.pos_ in allowed_postags]
    # for token in doc:
    #     print(token.text, token.pos_, token.dep_)
    # for ent in doc.ents:
    #     print(ent.text, ent.start_char, ent.end_char, ent.label_)
    return " ".join(tokens)
def lemm_en(text, use_ner=False, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN', 'INTJ', 'NUM', 'X']):
    nlp = nlp_en
    doc = nlp(text)
    tokens = [(lambda x: x.text if x.text in names else x.lemma_)(token) for token in doc if token.pos_ in allowed_postags]
    return " ".join(tokens)

## N-Grams / phrases

## Pré-traitement final

def preprocess_expe(docs, language, vocabulary_size=2000, make_ngrams=False, seuil_ngram = 25, min_count_ngram=10, max_df=0.95, min_df=3) :
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

  if make_ngrams :
    # Créer un texte à part pour les Ngrams à ce stade ?
    text_for_ngrams=[remove_hashtags(doc) for doc in preprocessed_docs_tmp]
    text_for_ngrams = [doc.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in text_for_ngrams]
    text_for_ngrams = [' '.join([w for w in doc.split() if len(w) > 0 and w not in stop_words]) for doc in text_for_ngrams]

  preprocessed_docs_tmp = [doc.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
  # Supprime whitespaces
  preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0]) for doc in preprocessed_docs_tmp]
  
  # On supprime les stop words, et aussi tout type de whitespace
  preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in stop_words]) for doc in preprocessed_docs_tmp]


    # LEMMATISATION : OPTIONNEL
  # if language == "russian":
  #   preprocessed_docs_tmp = [lemm_ru(doc) for doc in tqdm(preprocessed_docs_tmp)]
  # if language == "english":
  #   preprocessed_docs_tmp = [lemm_en(doc) for doc in tqdm(preprocessed_docs_tmp)]
  # lemmatized_docs = preprocessed_docs_tmp
  spacy_docs, preprocessed_docs_tmp=lemm(language, preprocessed_docs_tmp)


  # Extraction des mots les plus importants via matrice terme-document
  # AVANT ET APRES ???
  # On peut tester avec un tf-idf (pas retenu mais meilleur dans certains cas) :
  # tfidf = TfidfVectorizer(max_features=vocabulary_size)
  # Ou des n-grams
  # vectorizer = CountVectorizer(ngram_range=(1,3), max_features=vocabulary_size)
  removed={'before_ngrams':[], 'after_ngrams':[]}
  
  vectorizer = CountVectorizer(max_features=vocabulary_size, max_df=max_df, min_df=min_df)
  vectorizer.fit_transform(preprocessed_docs_tmp)
  vocabulary = set(vectorizer.get_feature_names_out())
  preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in vocabulary])
                                  for doc in preprocessed_docs_tmp]
  removed['before_ngrams']=vectorizer.stop_words_

  if make_ngrams : 
    bigram = gensim.models.Phrases([doc.split() for doc in preprocessed_docs_tmp], min_count=min_count_ngram, threshold=seuil_ngram, connector_words=connector_words, delimiter='_') # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[[doc.split() for doc in preprocessed_docs_tmp]], threshold=seuil_ngram, connector_words=connector_words, delimiter='_')  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    preprocessed_docs_tmp =[' '.join(trigram_mod[bigram_mod[doc.split()]]) for doc in preprocessed_docs_tmp]
    ngrams=trigram.export_phrases()
  
  # SECOND countvectorizer
  # juste pour obtenir le vocab
  vectorizer = CountVectorizer(max_features=None, max_df=1.0, min_df=1)
  vectorizer.fit_transform(preprocessed_docs_tmp)
  vocabulary = set(vectorizer.get_feature_names_out())
  
  preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in vocabulary])
                                  for doc in preprocessed_docs_tmp]

  vocab_out=vectorizer.get_feature_names_out()
  removed['after_ngrams']=vectorizer.stop_words_
  
  # ORIGINAL : ON CREE LES NGRAMS A PARTIR DU TEXTE PREPROC A CE STADE
  # if make_ngrams : 
  #   bigram = gensim.models.Phrases([remove_hashtags(doc).split() for doc in preprocessed_docs_tmp], min_count=10, threshold=25, connector_words=connector_words, delimiter='_') # higher threshold fewer phrases.
  #   trigram = gensim.models.Phrases(bigram[[remove_hashtags(doc).split() for doc in preprocessed_docs_tmp]], threshold=25, connector_words=connector_words, delimiter='_')  
  #   bigram_mod = gensim.models.phrases.Phraser(bigram)
  #   trigram_mod = gensim.models.phrases.Phraser(trigram)
  #   preprocessed_docs_tmp =[' '.join(trigram_mod[bigram_mod[remove_hashtags(doc).split()]]) for doc in preprocessed_docs_tmp]
  #   ngrams=trigram.export_phrases()
  
  preprocessed_docs, unpreprocessed_docs, deleted = [], [], []


  for i, doc in enumerate(preprocessed_docs_tmp):
    if len(doc) > 0:
      preprocessed_docs.append(doc)
      # Renvoie la liste des mêmes documents, non traités, pour BERT
      # pour les textes destinés à BERT on va quand même retirer les liens, les emoji... ?
      # unpreprocessed_docs.append(' '.join(remove_html(remove_emoji(docs[i]))).split())
      # test avec la fonction pour le modèle roberta-xlm-twitter
      unpreprocessed_docs.append(roberta_preprocess(docs[i]))
      # Renvoie une liste de booléens pour repérer les tweets supprimés (car vides) après preproc
      deleted.append(0)
    else :
      deleted.append(1)
  if make_ngrams :
    return removed, spacy_docs, ngrams, preprocessed_docs, unpreprocessed_docs, vocab_out, deleted
  else :
    return removed, spacy_docs, preprocessed_docs, unpreprocessed_docs, vocab_out, deleted

def preprocess(docs, language, vocabulary_size=2000, make_ngrams=False, seuil_ngram = 25, min_count_ngram=10, max_df=0.95, min_df=3) :
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

  # if make_ngrams :
  #   # Créer un texte à part pour les Ngrams à ce stade ?
  #   text_for_ngrams=[remove_hashtags(doc) for doc in preprocessed_docs_tmp]
  #   text_for_ngrams = [doc.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in text_for_ngrams]
  #   text_for_ngrams = [' '.join([w for w in doc.split() if len(w) > 0 and w not in stop_words]) for doc in text_for_ngrams]

  preprocessed_docs_tmp = [doc.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
  # Supprime whitespaces
  preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0]) for doc in preprocessed_docs_tmp]
  
  # On supprime les stop words, et aussi tout type de whitespace
  preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in stop_words]) for doc in preprocessed_docs_tmp]


    # LEMMATISATION : OPTIONNEL
  # if language == "russian":
  #   preprocessed_docs_tmp = [lemm_ru(doc) for doc in tqdm(preprocessed_docs_tmp)]
  # if language == "english":
  #   preprocessed_docs_tmp = [lemm_en(doc) for doc in tqdm(preprocessed_docs_tmp)]
  # lemmatized_docs = preprocessed_docs_tmp

  preprocessed_docs_tmp=[lemm(language, doc) for doc in tqdm(preprocessed_docs_tmp)]
  lemmatized_docs=preprocessed_docs_tmp


  # Extraction des mots les plus importants via matrice terme-document
  # AVANT ET APRES ???
  # On peut tester avec un tf-idf (pas retenu mais meilleur dans certains cas) :
  # tfidf = TfidfVectorizer(max_features=vocabulary_size)
  # Ou des n-grams
  # vectorizer = CountVectorizer(ngram_range=(1,3), max_features=vocabulary_size)
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
  
    # SECOND countvectorizer si on crée les ngrams
    # juste pour obtenir le nouveau vocab
    vectorizer = CountVectorizer(max_features=None, max_df=1.0, min_df=1)
    vectorizer.fit_transform(preprocessed_docs_tmp)
    vocabulary = vectorizer.get_feature_names()
  
    preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in set(vocabulary)])
                                    for doc in preprocessed_docs_tmp]

  vocab_out=list(vocabulary)
  removed['after_ngrams']=vectorizer.stop_words_
  
  # ORIGINAL : ON CREE LES NGRAMS A PARTIR DU TEXTE PREPROC A CE STADE
  # if make_ngrams : 
  #   bigram = gensim.models.Phrases([remove_hashtags(doc).split() for doc in preprocessed_docs_tmp], min_count=10, threshold=25, connector_words=connector_words, delimiter='_') # higher threshold fewer phrases.
  #   trigram = gensim.models.Phrases(bigram[[remove_hashtags(doc).split() for doc in preprocessed_docs_tmp]], threshold=25, connector_words=connector_words, delimiter='_')  
  #   bigram_mod = gensim.models.phrases.Phraser(bigram)
  #   trigram_mod = gensim.models.phrases.Phraser(trigram)
  #   preprocessed_docs_tmp =[' '.join(trigram_mod[bigram_mod[remove_hashtags(doc).split()]]) for doc in preprocessed_docs_tmp]
  #   ngrams=trigram.export_phrases()
  
  preprocessed_docs, unpreprocessed_docs, deleted = [], [], []


  for i, doc in enumerate(preprocessed_docs_tmp):
    if len(doc) > 0:
      preprocessed_docs.append(doc)
      # Renvoie la liste des mêmes documents, non traités, pour BERT
      # pour les textes destinés à BERT on va quand même retirer les liens, les emoji... ?
      # unpreprocessed_docs.append(' '.join(remove_html(remove_emoji(docs[i]))).split())
      # test avec la fonction pour le modèle roberta-xlm-twitter
      unpreprocessed_docs.append(roberta_preprocess(docs[i]))
      # Renvoie une liste de booléens pour repérer les tweets supprimés (car vides) après preproc
      deleted.append(0)
    else :
      deleted.append(1)
  if make_ngrams :
    return removed, lemmatized_docs, ngrams, preprocessed_docs, unpreprocessed_docs, vocab_out, deleted
  else :
    return removed, lemmatized_docs, preprocessed_docs, unpreprocessed_docs, vocab_out, deleted

# Fonctions séparées pour lemma, pré-traitement, vocabulary selection d'une liste de textes

def preprocess_lem(docs, language) :
  # On importe les stop_words de NLTK dans la langue idoine
  stop_words = stopwords.words(language)
  stop_words = set(stop_words)

  preprocessed_docs_tmp = docs
  # On retire la casse
  preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
  # On retire le html, les emojis, la ponctuation, stopwords
  preprocessed_docs_tmp = [remove_html(doc) for doc in preprocessed_docs_tmp]
  preprocessed_docs_tmp = [remove_emoji(doc) for doc in preprocessed_docs_tmp]
  preprocessed_docs_tmp = [remove_ats(doc) for doc in preprocessed_docs_tmp]
  preprocessed_docs_tmp = [doc.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
  # On supprime aussi tout type de whitespace
  preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in stop_words]) for doc in preprocessed_docs_tmp]
    # LEMMATISATION : OPTIONNEL
  if language == "russian":
    preprocessed_docs_tmp = [lemm_ru(doc) for doc in tqdm(preprocessed_docs_tmp)]
  if language == "english":
    preprocessed_docs_tmp = [lemm_en(doc) for doc in tqdm(preprocessed_docs_tmp)]

  return preprocessed_docs_tmp

def vectorize(docs, preprocessed_docs_tmp, vocabulary_size=10000) :
  # On choisit la taille du vocabulaire
    # Extraction des mots les plus importants via matrice terme-document
  # On peut tester avec un tf-idf
  # Ou des n-grams
  # vectorizer = CountVectorizer(ngram_range=(1,3), max_features=vocabulary_size)
  vectorizer = CountVectorizer(max_features=vocabulary_size, max_df=0.95, min_df=5)

  vectorizer.fit_transform(preprocessed_docs_tmp)
  vocabulary = set(vectorizer.get_feature_names())
  preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in vocabulary])
                                  for doc in preprocessed_docs_tmp]


  preprocessed_docs, unpreprocessed_docs, deleted = [], [], []


  for i, doc in enumerate(preprocessed_docs_tmp):
      if len(doc) > 0:
          preprocessed_docs.append(doc)
          # Renvoie la liste des mêmes documents, non traités, pour BERT
          # pour les textes destinés à BERT on va quand même retirer les liens, les emoji... ?
          # unpreprocessed_docs.append(' '.join(remove_html(remove_emoji(docs[i]))).split())
          # test avec la fonction pour le modèle roberta-xlm-twitter
          unpreprocessed_docs.append(roberta_preprocess(docs[i]))
          # Renvoie une liste de booléens pour repérer les tweets supprimés (car vides) après preproc
          deleted.append(0)
      else :
          deleted.append(1)

  return preprocessed_docs, unpreprocessed_docs, list(vocabulary), deleted

# def tfidf_vectorize(docs, preprocessed_docs_tmp) :
#   vocabulary_size=10000
#   vectorizer = TfidfVectorizer(max_features=vocabulary_size, max_df=0.95, min_df=5)

#   vectorizer.fit_transform(preprocessed_docs_tmp)
#   vocabulary = set(vectorizer.get_feature_names())
#   preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in vocabulary])
#                                   for doc in preprocessed_docs_tmp]


#   preprocessed_docs, unpreprocessed_docs, deleted = [], [], []


#   for i, doc in enumerate(preprocessed_docs_tmp):
#       if len(doc) > 0:
#           preprocessed_docs.append(doc)
#           # Renvoie la liste des mêmes documents, non traités, pour BERT
#           # pour les textes destinés à BERT on va quand même retirer les liens, les emoji... ?
#           # unpreprocessed_docs.append(' '.join(remove_html(remove_emoji(docs[i]))).split())
#           # test avec la fonction pour le modèle roberta-xlm-twitter
#           unpreprocessed_docs.append(roberta_preprocess(docs[i]))
#           # Renvoie une liste de booléens pour repérer les tweets supprimés (car vides) après preproc
#           deleted.append(0)
#       else :
#           deleted.append(1)

#   return preprocessed_docs, unpreprocessed_docs, list(vocabulary), deleted



import spacy
from spacy.tokens import DocBin, Doc
def add_docs_to_bin(texts, lang, attrs=["LOWER", "POS", "ENT_ID", "ENT_TYPE", "IS_ALPHA", "LEMMA", "LANG", "TAG"], to_disk_path=None):
    doc_bin = DocBin(attrs=attrs, store_user_data=True)
    doc_arrays=[]
    if lang=="english" :
        nlp=nlp_en
    else :
        nlp=nlp_ru
    for doc in tqdm(nlp.pipe(texts)):
        doc_bin.add(doc)
        doc_arrays.append(doc.to_array(attrs))
    bytes_data = doc_bin.to_bytes()
    if to_disk_path is not None :
        doc_bin.to_disk(to_disk_path)
    return(bytes_data, doc_arrays)
    

def light_preprocess(docs, language) :
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
  return(preprocessed_docs_tmp)

def preprocess_to_docbin(docs, language, to_disk_path=None) :
  
  # On supprime les stop words, et aussi tout type de whitespace
  preprocessed_docs_tmp = light_preprocess(docs, language)


    # LEMMATISATION : OPTIONNEL
  # if language == "russian":
  #   preprocessed_docs_tmp = [lemm_ru(doc) for doc in tqdm(preprocessed_docs_tmp)]
  # if language == "english":
  #   preprocessed_docs_tmp = [lemm_en(doc) for doc in tqdm(preprocessed_docs_tmp)]
  # lemmatized_docs = preprocessed_docs_tmp
  docbin_bytes, doc_arrays=add_docs_to_bin(preprocessed_docs_tmp, language, to_disk_path=to_disk_path)
    
  return (preprocessed_docs_tmp, docbin_bytes, doc_arrays)



# Deserialize later, e.g. in a new process
def read_spacy(data, language, attrs=["LOWER", "POS", "ENT_ID", "ENT_TYPE", "IS_ALPHA", "LEMMA", "LANG", "TAG"]) :
    nlp = spacy.blank(language[:2])
    if isinstance(data, bytes) :
        doc_bin = DocBin(attrs).from_bytes(data)
        docs = list(doc_bin.get_docs(nlp.vocab))
        return (docs)
    elif isinstance(data, list) :
        ldocs=[]
        for array in data :
            doc=Doc(nlp.vocab)
            ldocs.append(doc.from_array(attrs, array))


# Deserialize later, e.g. in a new process
def read_spacy_from_arrays(doc_arrays, texts, language, attrs=["LOWER", "POS", "ENT_ID", "ENT_TYPE", "IS_ALPHA", "LEMMA", "LANG", "TAG"]) :
    nlp = spacy.blank(language[:2])
    if isinstance(doc_arrays, list) :
        ldocs=[]
        for array, txt in zip(doc_arrays, light_preprocess(texts, language)) :
            doc=nlp(txt)
            doc2 = Doc(doc.vocab, words=[t.text for t in doc])
            ldocs.append(doc2.from_array(attrs, array))
        return (ldocs)

def run_parser() :
  for lang in  ['english', 'russian'] :
      data.loc[(data.lang == lang[:2]), 'light_preproc'], docbin_bytes, data.loc[(data.lang == lang[:2]), 'spacy_array'] = preprocess_to_docbin(data.loc[(data.lang == lang[:2])].text.to_list(), lang, to_disk_path='datasets/docs_'+lang[:2]+'.spacy')

  data.to_csv('datasets/beltweets_NLP_spacy.csv')