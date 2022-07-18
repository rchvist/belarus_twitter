# Do all models with pre-opti and multi
# library OCTIS
from octis.models.CTM import CTM
from octis.models.LDA import LDA
from octis.models.NMF import NMF
from octis.dataset.dataset import Dataset
from octis.optimization.optimizer import Optimizer
from skopt.space.space import Real, Categorical, Integer
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.models.model import save_model_output
import pyLDAvis as pdvs
import gensim.corpora as corpora
# libraries datascience, NLP
import pandas as pd
import numpy as np
import re
from shutil import rmtree
# Other, utilities
import os
import time
import pickle
from tqdm import tqdm
import shutil
from gensim.models import nmf, TfidfModel
import gensim.corpora as corpora
import gensim.corpora as corpora
import octis.configuration.citations as citations
import octis.configuration.defaults as defaults
from gensim.matutils import corpus2csc 
import numpy as np
from custom_models import NMF_TFIDF, LDA_TFIDF
import json
from nlp_preprocessing import make_TM_dataset
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# import sys

# if not sys.warnoptions:
#     import warnings
#     warnings.simplefilter("ignore")
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore",category=DeprecationWarning)
#     import scipy.optimize.linesearch, sklearn, pymorphy2, line_search_wolfe2
from nlp_preprocessing import gtranslate

# RUN MODELS

################################################## PARAMETER SELECTION #######################
infile="datasets/beltweets_nlp.csv"
start = "2020-08-06"
end = "2020-08-07"
split = "1D"
path='TM/MM_10k' 
data_path=path+'/datasets'# Where to store files for preprocessed dataset (mandatory to for OCTIS based models)
optimization_runs=25
model_runs=3
langs=['english', 'russian']
langs=['russian']

models = ['LDA'] # Possible choices :  models = ['NMF_tfidf', 'NMF',  'LDA', 'LDA_tfidf']

k_range=range(5,6) # Values of k (number of topics) to try
# Can be a custom list : 
# k_range=[5, 15, 25, 50]
# Or a single value 
k_range=[5, 6]
# Or fancier ranges
# k_range=[n for n in range(6,15,1)] + [n for n in range(15,33,3)]

# model_runs=2
# TODO For each run, run each model this many times, to account for random initialization of weights
# So far we don't do it for Vanilla CTM. We use fixed initialization of weights so we can load the model later.


##################################################

from octis.dataset.dataset import Dataset
for lang in langs :
    start_time=pd.Timestamp(start, tz='utc')
    end_time=pd.Timestamp(end, tz='utc')
    if split==None :
        split=end_time-start_time
    else :
        split=pd.Timedelta(split)
    start_str="".join(start.split("-"))
    end_str="".join(end.split("-"))
    while start_time < end_time :
        print (f"Preprocessing {lang} data from {start} to {(start_time+split).strftime('%Y-%m-%d')}, out of {end})")
        preprocessed, unpreprocessed, vocab, deleted=make_TM_dataset(
        infile=infile, start=start_time, end=start_time+pd.Timedelta(split), lang=lang, 
        save_path=data_path, vocabulary_size=10000, make_ngrams=False, 
        max_df=0.95, min_df=2, seuil_ngram=8, min_count_ngram=2)
        
        for m in models :
            dataset = Dataset()
            dataset.load_custom_dataset_from_folder(data_path+f'/octis_preprocessed/{start_str}_{end_str}_{lang}')

            # Metrics
            c_v = Coherence(texts=dataset.get_corpus(), measure = 'c_v')
            npmi = Coherence(texts=dataset.get_corpus(), topk=10, measure='c_npmi')
            topic_diversity = TopicDiversity(topk=10)
            extra_metrics=[npmi, topic_diversity]
            
            # Optimize alpha parameter w/ 11 topics before running for LDA models
            search_space={}
            print(f'Training {lang} models for period {start} to {end}.' )
            if 'LDA' in m :
                print('Optimizing LDA alpha...')
                if m=='LDA' :
                    opti_model = LDA(num_topics=11)
                elif m=='LDA_tfidf' :
                    opti_model = LDA_TFIDF(num_topics=11)
                search_space["alpha"]= Real(low=0.02,high= 0.2)
                optimizer=Optimizer()
                optimization_results = optimizer.optimize( # res= optimization result (json like)
                opti_model, dataset, c_v, search_space, number_of_call=optimization_runs, 
                model_runs=1, # we only want the best run
                save_models=False, # we will not save them all for this pre-optimization
                extra_metrics=None, # neither do we keep track of other metrics
                save_path=path+'/pre_opti_results/')
                res=json.load(open(path+'/pre_opti_results/result.json','r'))
                optimal_iter = res['f_val'].index(max(res['f_val']))
                    # for now we only opti 1 parameter but we still load them as a list
                optimal_params = [res['x_iters'][parameter][optimal_iter] for parameter in res['x_iters'].keys()]
                best_alpha=optimal_params[0]
                print('pre-optimization done, best iter for model %s was %d, with params : %s' %(m, optimal_iter, str(optimal_params)))
                rmtree(path+'/pre_opti_results')
            
            # Custom run and evaluate for different n_topics
            results={'n_topics':[], 'coherence':[], 'diversity':[]} # results for custom benchmark of n_topics 
            for k in k_range :
                if m =='NMF' :
                    model = NMF(use_partitions=False, num_topics=k)
                elif m =='NMF_tfidf' :
                    model = NMF_TFIDF(use_partitions=False, num_topics=k)
                elif m=='LDA' :
                    model = LDA(alpha=best_alpha, num_topics=(k))
                elif m=='LDA_tfidf' :
                    model = LDA_TFIDF(alpha=best_alpha, num_topics=k)

                for n in range(model_runs) :
                    # load models
                    if 'LDA' in m :
                        output=model.train_model(dataset)
                    else :
                        output=model.train_model(dataset, top_words=15)
                    results['n_topics'].append(k)
                    results['coherence'].append(c_v.score(output))
                    results['diversity'].append(topic_diversity.score(output))
                
                    name=f'{m}_{k}_{n}'
                    models_folder=path+f'/models/{start_str}_{end_str}_{lang}'+'/models'
                    os.makedirs(models_folder, exist_ok=True)
                    save_model_output(model_output=output, path=models_folder+'/'+name)
                    print('done, output : '+path+'/models/'+name)
            pd.DataFrame(results).to_csv(path+f'/{start}_{end}_{lang}_{m}_results.csv')

        ######### MAKE PYLDAVIS, CHECK TOPICS #################
        print('Checking best model...')
        topics={}
        totranslate={}
        tocompare = []
        for l in langs :
            tocompare.append(models)

        for u,lang in enumerate(langs) :
            topics[lang]={}
            for i,m in enumerate(tocompare[u]) :
                
                df = pd.read_csv(path+f'/{start}_{end}_{lang}_{m}_results.csv', index_col=0).set_index('n_topics')
                folder=m

                # Sélection auto
                norm_df = ((df-df.mean())/df.std())
                
                # 1er n_topics après que les courbes se croisent (sila topic diversity décroit!)
                if norm_df['diversity'].iloc[0]>norm_df['diversity'].iloc[-1] :
                    optimal_iter=norm_df[norm_df['coherence']>=norm_df['diversity']].iloc[0].name
                    optimal_run=df.loc[optimal_iter].reset_index()['coherence'].idxmax()
                else :
                    optimal_iter=norm_df.groupby(norm_df.index)['coherence'].median().idxmax()
                    optimal_run=df.loc[optimal_iter].reset_index()['coherence'].idxmax()                

                with np.load(models_folder+f"/{m}_{optimal_iter}_{optimal_run}.npz") as model :
                    if m=="NMF" and lang =='russian' :
                        print("Loading iteration : "+"%s/models/%d_%d.npz" % (folder+m+'/'+lang, optimal_iter-min(df.index), optimal_run))
                    tops = model['topics']
                    topic_term=model['topic-word-matrix']
                    topic_doc=model['topic-document-matrix']
                    topics[lang][m]={}
                    topics[lang][m]['topics']=tops
                    topics[lang][m]['topic_term']=topic_term
                    topics[lang][m]['topic_doc']=topic_doc
                
                id2word = corpora.Dictionary(dataset.get_corpus())
                doc_lengths=[len(tw) for tw in dataset.get_corpus()]
                viz_data=pdvs.prepare(topic_term, np.transpose(topic_doc), doc_lengths, [id2word[i] for i in range(len(topic_term.T))] , id2word.cfs.values())
                pdvs.save_html(viz_data, path+f'/{start}_{end}_{lang}_{m}_pyldavis.html')
                
                print(lang.upper()+'________'+m+' : ' + str(len(tops)) + ' topics found_______________________________________')    
                # Do something with the topics
                if lang== 'russian' :
                    [(print (*t), print(gtranslate(t))) for t in tops]
                else :
                    [print (*t) for t in tops]

        start_time+=split