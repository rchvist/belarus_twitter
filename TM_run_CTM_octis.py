# Preprocessing functions
from genericpath import isdir, exists
from nlp_preprocessing import make_TM_dataset
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
# libraries datascience
import pandas as pd
import numpy as np
# other NLP
import pyLDAvis as pdvs
import gensim.corpora as corpora
# Other, utilities
import os
import shutil
from gensim.models import nmf, TfidfModel
import gensim.corpora as corpora
import gensim.corpora as corpora
import octis.configuration.citations as citations
import octis.configuration.defaults as defaults
from gensim.matutils import corpus2csc 
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

################################################## PARAMETER SELECTION #######################
infile="datasets/beltweets_nlp.csv"
start = "2020-08-09"
end = "2020-08-09"
# split = '30D'
path='TM/B7_2k_095_POS' 
data_path=path+'/dataset'# OCTIS Topic models require pre-processed data to be stored in a folder in a specific format. We will create and save a sub-dataset here

lang='english'
# If we want to run several base berts (LLM Component of CTM, see https://aclanthology.org/2021.acl-short.96/)
# models = ['cardiffnlp/twitter-roberta-base', 'bert-base-nli-mean-tokens', "paraphrase-distilroberta-base-v2"]
models = ['cardiffnlp/twitter-roberta-base']
# lang = 'russian'
# models = ['cardiffnlp/twitter-xlm-roberta-base','cointegrated/rubert-tiny2', 'DeepPavlov/rubert-base-cased-sentence']

# Values of k (number of topics) to try
k_range=range(3,8)
# Can be a custom list : 
# k_range=[5, 15, 25, 50]
# Or a single value 
# k_range=[8]
# Or fancier ranges
# k_range=[n for n in range(6,15,1)] + [n for n in range(15,33,3)]

# Run each model this many times, to account for random initialization of weights
model_runs=3

##################################################

preprocessed, unpreprocessed, vocab, deleted=make_TM_dataset(
    infile=infile, start=start, end=end, lang=lang, save_path=data_path, vocabulary_size=2000, make_ngrams=False, max_df=0.7, min_df=2, seuil_ngram=8, min_count_ngram=2)

print('starting...')

# for f_date in os.scandir('datasets/CTM_2k/preprocessed') :
# TO ITERATE OVER SEVERAL SPLITS
for f_date in [fol for fol in os.scandir(data_path+'/octis_preprocessed') if isdir(fol)] :
    i=f_date.name
    print ('i='+str(i))
    # octis_path = data_path+"/octis_preprocessed/{start}_{end}_{lang}"
    print('loading dataset from '+i+'...')
    for num_model,m in enumerate(models) :
        results={'n_topics':[], 'coherence':[], 'diversity':[]}
        dataset = Dataset()
        dataset.load_custom_dataset_from_folder(f_date.path)
        # Metrics
        c_v = Coherence(texts=dataset.get_corpus(), measure = 'c_v')
        npmi = Coherence(texts=dataset.get_corpus(), topk=10, measure='c_npmi')
        topic_diversity = TopicDiversity(topk=10)
        extra_metrics=[npmi, topic_diversity]
        # Evaluate for different topic values
        save_path=path+'/%s/%s_%s' %(str(i), lang, m.split('/')[-1])
        bert_path=save_path+'/base_bert/'
        for k in (k_range) :
            for n in range(model_runs) :
                os.makedirs(bert_path, exist_ok=True)
                model = CTM(
                    num_epochs=40, dropout = 0.09891608522984201, num_neurons=300, inference_type='combined',
                    use_partitions=False, bert_model=m, bert_path=bert_path, num_topics=k)
                # load models
                output=model.train_model(dataset, top_words=15)
                results['n_topics'].append(k)
                results['coherence'].append(c_v.score(output))
                results['diversity'].append(topic_diversity.score(output))
                os.makedirs(save_path+'/models/', exist_ok=True)
                m_name=str(k)+'_'+str(n)+"_m"+str(num_model)
                save_model_output(model_output=output, path=save_path+'/models/'+m_name)
                print('finished run : '+save_path+'/models/'+m_name)
        df=pd.DataFrame(results)
        df.to_csv(path+'/results.csv')
        print(f'model run finished for {m}, getting topics out')


        ######### FROM CHECKTOPICS_OCTIS.PY
        # Auto select best K and output topics
        topics={}
        df=df.set_index('n_topics')
        norm_df = ((df-df.mean())/df.std())
        folder=path
        # 1er n_topics après que les courbes se croisent (sila topic diversity décroit!)
        if norm_df['diversity'].iloc[0]>norm_df['diversity'].iloc[-1] :
            optimal_iter=norm_df[norm_df['coherence']>=norm_df['diversity']].iloc[0].name
            optimal_run=df.loc[optimal_iter].reset_index()['coherence'].idxmax()
        else :
            optimal_iter=norm_df.groupby(norm_df.index)['coherence'].median().idxmax()
            optimal_run=df.loc[optimal_iter].reset_index()['coherence'].idxmax()
        # Get the complete bert model name (cointegrated/... etc)
        bertmodel=m
        m=m.split('/')[-1]
        # with np.load("%s/models/%d_%s.npz" % (folder+lang+'_'+m, optimal_iter, 'm'+str(i))) as model :
        # If we did several runs
        
        optimal_run=df.loc[optimal_iter].reset_index()['coherence'].idxmax()
        with np.load("%s/models/%d_%d_%s.npz" % (save_path, optimal_iter, optimal_run, 'm'+str(num_model))) as model :
            tops = model['topics']
            topic_term=model['topic-word-matrix']
            topic_doc=model['topic-document-matrix']

        # FOR CTM MODEL GET THE SOFTMAX OF THE TOPIC_TERM MATRIX, THEN NORMALIZE IT
        # Prepare dataset and model for pyldavis
        from scipy.special import softmax
        from sklearn.preprocessing import normalize
        dataset = Dataset()
        dataset.load_custom_dataset_from_folder(data_path+f"/octis_preprocessed/{''.join(start.split('-'))}_{''.join(end.split('-'))}_{lang}")
        base_bert_path=bert_path+'/_train.pkl'
        # retrain embeddings to get the vocab in right order....
        preprocess = CTM.preprocess
        ctmdataset, input_size=preprocess(dataset.get_vocabulary(), train=[' '.join(i) for i in dataset.get_corpus()], bert_train_path=base_bert_path, bert_model=bertmodel)
        idx2token = ctmdataset.idx2token

        counts= ctmdataset.X.sum(axis=0)

        doc_lengths=[len(tw) for tw in dataset.get_corpus()]
        viz_data=pdvs.prepare(normalize(softmax(topic_term, axis=1), axis=1, norm='l1'), np.transpose(topic_doc), doc_lengths, idx2token, counts)

        pdvs.save_html(viz_data, folder+i+'_pyldavis.html')
        print(lang.upper()+'__________'+m+'________________________________________________')
        print (f'{i}, {optimal_iter} topics found')
        # Do something with the topics
        [print (*t) for t in tops]
# [print(m, '\n___', gtranslate(t)) for t in [tops for tops in [topics['russian'][m]['topics'] for m in topics['russian']]]]