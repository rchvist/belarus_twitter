# library OCTIS
from octis.models.CTM import CTM
from octis.models.LDA import LDA
from octis.dataset.dataset import Dataset
from octis.optimization.optimizer import Optimizer
from skopt.space.space import Real, Categorical, Integer
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
# libraries datascience, NLP
import pandas as pd
import numpy as np
import re
# Other, utilities
import os
import time
import pickle
from tqdm import tqdm
import shutil
print('sleeping 1h')
time.sleep(3600)
print('starting')
optimization_runs=25
model_runs=3
start=time.time()
print('starting :' + str(start))
from octis.dataset.dataset import Dataset
for lang in ['english'] :
    for i in ["aug08_aug31"] :
        search_space = {"num_topics": Categorical({15, 20, 26, 28, 30, 32, 34, 35, 40})}
        dataset = Dataset()
        dataset.load_custom_dataset_from_folder("bench/LDA/%s/%s" %(i, lang))
        # Metrics
        c_v = Coherence(texts=dataset.get_corpus(), measure = 'c_v')
        npmi = Coherence(texts=dataset.get_corpus(), topk=10, measure='c_npmi')
        umass=Coherence(texts=dataset.get_corpus(), topk=10, measure='c_umass')
        topic_diversity = TopicDiversity(topk=10)
        extra_metrics=[npmi, c_v, topic_diversity]
        # Load CTM Data and model
        # TODO Should we use preprocessed/tokenized tweets 
        if lang== 'english' :
            os.makedirs('bench/base_berts_en/CTM3_%s_%s/'%((i), lang))
            search_space['bert_model']= Categorical({'cardiffnlp/twitter-xlm-roberta-base', 'cardiffnlp/twitter-roberta-base','bert-base-nli-mean-tokens'})

            CTMmodel = CTM(num_epochs=50, inference_type='combined', use_partitions=False, bert_path='bench/base_berts_en/CTM3_%s_%s/'%((i), lang))
            optimizer_ctm=Optimizer()
        else : 
            os.makedirs('bench/base_berts_ru/CTM3_%s_%s/'%((i), lang))
            search_space['bert_model']= Categorical({'cardiffnlp/twitter-xlm-roberta-base','cointegrated/rubert-tiny2', 'DeepPavlov/rubert-base-cased-sentence'})
            CTMmodel = CTM(num_epochs=50, inference_type='combined', use_partitions=False, bert_path='bench/base_berts_ru/CTM3_%s_%s/'%((i), lang))
            optimizer_ctm=Optimizer()
            # Optimize CTM
        print('Running CTM benchmark for %s %s' %(i, lang))
        optimization_result_ = optimizer_ctm.optimize(
        CTMmodel, dataset, umass, search_space, number_of_call=optimization_runs, 
        model_runs=model_runs, save_models=True, 
        extra_metrics=extra_metrics, # to keep track of other metrics
        save_path='bench/result_ctm3_%s_%s/'%((i), lang))
print("done benchmarking on preprocessed corpus")
        
end = time.time()
print(str(end))