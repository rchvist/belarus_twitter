# Preprocessing functions
from genericpath import isdir, exists
from nlp_preprocessing import make_TM_dataset
#library CTM
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file, TopicModelDataPreparation
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.evaluation.measures import CoherenceNPMI, InvertedRBO, CoherenceCV, TopicDiversity
from gensim.corpora.dictionary import Dictionary
from gensim.models import ldamodel 
import pyLDAvis

#libraries datascience
import pandas as pd
import os
import numpy as np
import pickle
from tqdm import tqdm
#autres librairies nlp (pre processing...)
import nltk
import gensim
from string import punctuation
from bs4 import BeautifulSoup
import re
#Preprocess function
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
import warnings
from nltk.corpus import stopwords
# Run the vanilla CTM on B7 (issue with B8 which uses the phraser)
import random
import numpy as np
import torch
batch_size=32
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

################################################## PARAMETER SELECTION #######################
infile="datasets/beltweets_nlp.csv"
start = "2020-08-06"
end = "2020-08-08"
split = "1D"
path='TM/CTM_vanilla' 
data_path=path+'/datasets'# Where to store files for preprocessed dataset (mandatory to for OCTIS based models)

langs=['english']
# If we want to run several base berts (LLM Component of CTM, see https://aclanthology.org/2021.acl-short.96/)
# models = ['cardiffnlp/twitter-roberta-base', 'bert-base-nli-mean-tokens', "paraphrase-distilroberta-base-v2"]
models = ['cardiffnlp/twitter-roberta-base']
# lang = 'russian'
# models = ['cardiffnlp/twitter-xlm-roberta-base','cointegrated/rubert-tiny2', 'DeepPavlov/rubert-base-cased-sentence']
num_epochs=5
# Values of k (number of topics) to try
k_range=range(3,8)
# Can be a custom list : 
# k_range=[5, 15, 25, 50]
# Or a single value 
k_range=[5, 6]
# Or fancier ranges
# k_range=[n for n in range(6,15,1)] + [n for n in range(15,33,3)]

# TODO Run each model this many times, to account for random initialization of weights
# So far we don't do it for Vanilla CTM. We use fixed initialization of weights so we can load the model later.
# model_runs=2

##################################################
for lang in langs :
    try :
        
    # Training CTM
        
        # results={'russian':{'topics' : [], 'models':[], 'topic_distrib':[]}, 'english':{'topics' : [], 'models':[], 'topic_distrib':[]}}
    # import torch
    # from transformers import AutoTokenizer, AutoModel
    # tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    # model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    # model.cuda()
        
        if lang=='russian' :
            # qt=TopicModelDataPreparation("cointegrated/rubert-tiny2")
            qt=TopicModelDataPreparation("DeepPavlov/rubert-base-cased-sentence")
        if lang=='english' :
            qt=TopicModelDataPreparation("cardiffnlp/twitter-roberta-base")
        start=pd.Timestamp(start, tz='utc')
        end=pd.Timestamp(end, tz='utc')
        if split==None :
            split=end-start
        else :
            split=pd.Timedelta(split)

        while start < end :
            
            results={'topics' : [], 'topic_distrib':[], 'coherence':[], 'diversity':[]}
            print (f'Preprocessing {lang} data from {start} to {start+split}, (end = {end})')
            preprocessed, unpreprocessed, vocab, deleted=make_TM_dataset(
            infile=infile, start=start, end=start+pd.Timedelta(split), lang=lang, save_path=data_path, vocabulary_size=2000, make_ngrams=False, max_df=0.7, min_df=2, seuil_ngram=8, min_count_ngram=2)
            a=[t for i,t in enumerate(preprocessed) if ~deleted[i]]
            b=[t for i,t in enumerate(unpreprocessed) if ~deleted[i]]

            # if not exists(data_path+f'/{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}_{lang}.csv') :
            #     print (f'Preprocessing {lang} data from {start.strftime("%Y%m%d")} to {start+split}, (end = {end.strftime("%Y%m%d")})')
            #     preprocessed, unpreprocessed, vocab, deleted=make_TM_dataset(
            #     infile=infile, start=start, end=start+pd.Timedelta(split), lang=lang, save_path=data_path, vocabulary_size=2000, make_ngrams=False, max_df=0.7, min_df=2, seuil_ngram=8, min_count_ngram=2)
            #     a=[t for i,t in enumerate(preprocessed) if ~deleted[i]]
            #     b=[t for i,t in enumerate(unpreprocessed) if ~deleted[i]]
            #     c=vocab
            # else :
            # # Use this if you want to make topics from files saved from make_TM_dataset
            #     data=pd.read_csv(data_path+f'/{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}_{lang}.csv')
            #     with open(data_path+f'/{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}_{lang}/octis_preprocessed/vocabulary.txt') as file:
            #         lines = [line.rstrip() for line in file]

                # # a = preprocessed, b= unpreprocessed, feed them in this order
                # (a, b,c) = (data[~data.deleted]['preproc'].to_list(), 
                #             data[~data.deleted]['unpreproc'].to_list(),
                #             lines)

            # To deal with a bug when Torch is fed a batch of size 1
            while len(a)%batch_size==1 :
                batch_size+=batch_size
            print('batch size : ', batch_size)

            print(f'Making base Bert dataset for {lang}...')
            training=qt.fit(b, a)
            print("training.X_bow.shape = ", training.X_bow.shape,
                "training.X_contextual.shape =", training.X_contextual.shape,
                )
            # make a few values of n_topics  :
            for k in k_range :
                print(f'training {lang} model for period {start.strftime("%Y%m%d")} to {end.strftime("%Y%m%d")} : {k} topics')
                torch.manual_seed(10)
                torch.cuda.manual_seed(10)
                np.random.seed(10)
                random.seed(10)
                torch.backends.cudnn.enabled = False
                torch.backends.cudnn.deterministic = True
                # ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=k, num_epochs=30) # base was 15
                # ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=k, num_epochs=50, batch_size=batch_size) # base was 15
                ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=k, num_epochs=num_epochs, batch_size=batch_size) # base was 15
                print(f'fitting {lang} model for data from {start.strftime("%Y%m%d")} to {start+split}')
                ctm.fit(training, n_samples=5)
                # ctm.fit(training)
                # path = path+"bench/results/B7_2k_095_POS/CTM_vanilla/%s/%d/"%(lang, (i))
                # path = path+"/%s/%d/"%(lang, (i))
                ctm.save(models_dir=path+f'/models/{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}_{lang}_{k}topics')
                # results['models'].append(ctm)
                print('getting topic list....')
                # Output topics in separate CSV
                topicslist=ctm.get_topic_lists(10)
                topics={i:t for (i, t) in enumerate(topicslist)}
                topics=pd.Series(topics)
                print('getting topic distrib for {k}')
                topics.to_csv(path+f'/{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}_{lang}_{k}topics.csv')
                # Make a dict of results
                results['topics'].append(topicslist)
                print("results['topics']",results['topics'])
                tweet_topics = ctm.get_doc_topic_distribution(training, n_samples=5)
                results['topic_distrib'].append(tweet_topics)
                print("results['topic_distrib']",results['topic_distrib'])
                coherence=CoherenceCV(topics=topicslist, texts= [t.split() for t in a])
                results['coherence'].append(coherence.score())
                print("results['coherence']",results['coherence'])
                diversity=TopicDiversity(topics=ctm.get_topic_lists(25))
                results['diversity'].append(diversity.score())
                print("results['diversity']",results['diversity'])
            print('lengths of keys : ', [(k, len(results[k])) for k in list(results.keys())])
            

            ############### GET PYLDAVIS.... #########################
            # Taken from B7_checktopics_en
            # datapath="bench/orig_datasets/B7_2k_095_POS/"
            # resultpath='bench/results/B7_2k_095_POS/CTM_vanilla/'

            #Check vocab file
            # vocab_path=f"datasets/B7_2k_095_POS/{d}/{lang}/vocabulary_n.txt"
            # with open(vocab_path) as file:
                # lines = [line.rstrip() for line in file]
            # vocab_fromfile=lines
            
            # Chose model (best K value) to load
            # if lang=="russian":
            #     optimal_iter=[4,4,6,7,4][i]
            # else:
            #     optimal_iter=5

            res_df=pd.DataFrame(results, index=k_range)
            norm_df=res_df[['coherence', 'diversity']]
            norm_df = ((norm_df-norm_df.mean())/norm_df.std())

            # If it is a model with a trade-off between diversity and coherence, get the lowest value of K after where the curves intersect
            if (norm_df['diversity'].iloc[0]>norm_df['diversity'].iloc[-1]) and (norm_df['coherence'].iloc[0]<norm_df['diversity'].iloc[-1]):

                optimal_iter=norm_df[norm_df['coherence']>=norm_df['diversity']].iloc[0].name
                # TODO allow for runs
                # optimal_run=res_df.loc[optimal_iter].reset_index()['coherence'].idxmax()
            else :
                # print(norm_df[norm_df['coherence']>=norm_df['diversity']].iloc[0])
                optimal_iter=norm_df.groupby(norm_df.index)['coherence'].median().idxmax()
                # TODO allow for runs
                # optimal_run=res_df.loc[optimal_iter].reset_index()['coherence'].idxmax()


            # topics = pd.read_csv(resultpath+f'{lang}/{d}/topics_{lang}_{d}_{optimal_iter}.csv', lineterminator='\n')
            topics = res_df['topics'].loc[optimal_iter]
    # Do something with the topics
            [print(i, t) for i, t in enumerate(topics)]
            # print(f'retraining embeddings for {lang}, {d}')
            # # Retrain the embeddings
            # data=pd.read_csv(datapath+f'{d}_{lang}.csv',  lineterminator='\n')

            # (a, b) = (data[~data.deleted]['preproc'].to_list(), 
            #     data[~data.deleted]['unpreproc'].to_list(),
            #     )
            # training = qt.fit(b, a)
            vocab_fromtraining = qt.vocab
            # print(f'loading model for {lang}, {d}')
            # Load "best k" model
            # ctm = CombinedTM(bow_size=len(vocab_fromtraining), contextual_size=768, n_components=30, num_epochs=50)
            modelfolder= [f for f in os.scandir(path+f'/models/{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}_{lang}_{optimal_iter}topics')][0] # we load the optimal model from the folder with a v long name
            ctm.load(modelfolder.path, epoch=num_epochs-1)
            print(f'loading best {lang} model for, {optimal_iter} topics and making PyLDAVis')
            
            lda=ctm.get_ldavis_data_format(vocab_fromtraining, training, 10)
            vis_data1 = pyLDAvis.prepare(**lda)
            pyLDAvis.save_html(vis_data1, path+f'/{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}_{lang}_{optimal_iter}topics.html')

            start+=split

    except ValueError as v:
        print(v)
        print('batch size : ', batch_size)
        print('len texts', len(a))
        print(len(a)%batch_size)
        print(batch_size%2)
        