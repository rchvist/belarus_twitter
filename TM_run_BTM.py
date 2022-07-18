import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import bitermplus as btm
import numpy as np
import pandas as pd
import pickle as pkl

from deep_translator import GoogleTranslator
from deep_translator.exceptions import NotValidPayload


def gtranslate(words, src='auto', lang='en') :
    res=[]
    for w in words :
        try :
            translated = GoogleTranslator(source=src, target=lang).translate(w)
            res.append(translated)
        except NotValidPayload :
            res.append(w)
    return(res)

# IMPORT FONCTION DE PRE PROCESS D'ORIGINE
import re
def roberta_preprocess(text):
    new_text = []
    for t in text.split():
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', 'http', str(t))
        new_text.append(t)
    return " ".join(new_text)
# pr sélection des docs, cf paper :
                            # To reduce lowquality
                            # tweets, we processed the raw content via the following
                            # normalization steps: (a) removing non-Latin characters
                            # and stop words; (b) converting letters into lower case; (c)
                            # removing words with document frequency less than 10; (d)
                            # filtering out tweets with length less than 2; (e) removing duplicate
                            # tweets.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
def vectorize(docs, preprocessed_docs_tmp) :
    # On choisit la taille du vocabulaire
    vocabulary_size=10000
    # Extraction des mots les plus importants via matrice terme-document
    # On peut tester avec un tf-idf
    # Ou des n-grams
    # vectorizer = CountVectorizer(ngram_range=(1,3), max_features=vocabulary_size)
    vectorizer = CountVectorizer(max_features=vocabulary_size, max_df=0.99, min_df=10)

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


# MODEL RUNS
langs = ['russian', 'english']
results=dict.fromkeys(langs)
model_runs = 3
for lang in langs :
    # IMPORTING DATA
    results[lang]={}
    dt=pd.read_csv('bench/orig_datasets/LDA_10k_095_POS/aug01_31_' + lang + '.csv', lineterminator='\n')
    dt=dt.dropna(subset=["lemmatized_POS"])
    dt=dt.join(pd.read_csv('datasets/LDA_10k_095_POS/beltweets_nlp.csv').set_index('status_id')['created_at'], on='status_id')
    dt=dt.set_index(pd.to_datetime(dt['created_at'])).sort_index()
    date=pd.Timestamp(year=2020, month=8, day=1)
    for day in range(8) :
        df=dt[df.index.date == date]
        preprocessed_docs, unpreprocessed_docs, vocab, deleted = vectorize(df['text'].str.strip().to_list(), df['lemmatized_POS'].str.strip().to_list())
        df['deleted']=deleted
        df['deleted']=df['deleted'].astype(bool)
        df=df[[c for c in df.columns if c not in ['preproc', 'unpreproc']]]
        df.loc[~df.deleted, 'preproc']=preprocessed_docs
        df.loc[~df.deleted, 'unpreproc']=unpreprocessed_docs
        df=df.drop_duplicates(subset=['preproc'], keep='first')
        texts=df[~df.deleted]['preproc'].str.strip().tolist()
        # IMPORT DATA FROM OCTIS .TSV
        # df = pd.read_csv(
        #     f'datasets/LDA_10k_095_POS/aug01_31/{lang}/corpus.tsv', sep='\t', header=None)
        # # df=df.drop_duplicates(keep='first')
        # texts = df[0].str.strip().tolist()
        #
        results[lang]['orig_df']=df
        results[lang]['texts']=texts
        # PREPROCESSING
        # Obtaining terms frequency in a sparse matrix and corpus vocabulary
        X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
        tf = np.array(X.sum(axis=0)).ravel()
        # Vectorizing documents
        docs_vec = btm.get_vectorized_docs(texts, vocabulary)
        docs_lens = list(map(len, docs_vec))
        # Generating biterms
        biterms = btm.get_biterms(docs_vec)
        # Creating dict for the metrics
        metrics = {'n_topics':[], 'coherence':[], 'perplexity':[]}
        results[lang]['models']=[]
        for i, n_topics in enumerate(range (3,13)) :
            for r in range (model_runs) :
                print(f'run {r} for {n_topics} topics')
                # INITIALIZING AND RUNNING MODEL
                # model = btm.BTM(
                #     X, vocabulary, seed=12321, T=n_topics, M=20, alpha=50/n_topics, beta=0.01, has_background=True)
                model = btm.BTM(
                    X, vocabulary, T=n_topics, M=20, alpha=50/n_topics, beta=0.01, has_background=True)
                model.fit(biterms, iterations=50)
                p_zd = model.transform(docs_vec)
                # METRICS
                perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, 8)
                coherence = btm.coherence(model.matrix_topics_words_, X, M=10)
                metrics['n_topics'].append(n_topics)
                metrics['coherence'].append(coherence)
                metrics['perplexity'].append(perplexity)
                results[lang]['models'].append(model)
                with open(f"bench/results/B7_10k_095_POS/Biterm/{lang}/models/{i}_{r}.pkl", "wb") as file:
                    pkl.dump(model, file)
        date = date + pd.DateOffset(1)

    results[lang]['metrics']=metrics
    metrics_df=pd.DataFrame(metrics)
    metrics_df.to_csv(f"bench/results/B7_10k_095_POS/Biterm/{lang}/results.csv")