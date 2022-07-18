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
from gensim.models import nmf, TfidfModel
import gensim.corpora as corpora
import gensim.corpora as corpora
import octis.configuration.citations as citations
import octis.configuration.defaults as defaults
from gensim.matutils import corpus2csc 
import numpy as np

### IMPORTS FOR OCTIS LDA
from octis.models.model import AbstractModel
import numpy as np
from gensim.models import ldamodel
import gensim.corpora as corpora
import octis.configuration.citations as citations
import octis.configuration.defaults as defaults

### DEFINE NEW MODEL FOR NMF, TF-IDF
class NMF_TFIDF(NMF) :
    def train_model(self, dataset, hyperparameters=None, top_words=10):
        if hyperparameters is None:
            hyperparameters = {}
        else :
            print(hyperparameters)
        if self.use_partitions:
            partition = dataset.get_partitioned_corpus(use_validation=False)
        else:
            partition = [dataset.get_corpus(), []]
        if self.id2word is None:
            self.id2word = corpora.Dictionary(dataset.get_corpus())
        if self.id_corpus is None:
            self.id_corpus = [self.id2word.doc2bow(
                document) for document in partition[0]]
            self.tfidf = TfidfModel(self.id_corpus, smartirs='ntc')
            self.tfidfcorpus=corpus2csc(self.tfidf[self.id_corpus])
        hyperparameters["corpus"] = self.tfidfcorpus
        hyperparameters["id2word"] = self.id2word
        self.hyperparameters.update(hyperparameters)
        
        self.trained_model = nmf.Nmf(**self.hyperparameters)

        result = {}

        result["topic-word-matrix"] = self.trained_model.get_topics()

        if top_words > 0:
            topics_output = []
            for topic in result["topic-word-matrix"]:
                top_k = np.argsort(topic)[-top_words:]
                top_k_words = list(reversed([self.id2word[i] for i in top_k]))
                topics_output.append(top_k_words)
            result["topics"] = topics_output

        result["topic-document-matrix"] = self._get_topic_document_matrix()

        if self.use_partitions:
            new_corpus = [self.id2word.doc2bow(
                document) for document in partition[1]]
            if self.update_with_test:
                self.trained_model.update(new_corpus)
                self.id_corpus.extend(new_corpus)

                result["test-topic-word-matrix"] = self.trained_model.get_topics()

                if top_words > 0:
                    topics_output = []
                    for topic in result["test-topic-word-matrix"]:
                        top_k = np.argsort(topic)[-top_words:]
                        top_k_words = list(
                            reversed([self.id2word[i] for i in top_k]))
                        topics_output.append(top_k_words)
                    result["test-topics"] = topics_output

                result["test-topic-document-matrix"] = self._get_topic_document_matrix()
            else:
                result["test-topic-document-matrix"] = self._get_topic_document_matrix(new_corpus)
        return result

### DEFINE NEW MODEL FOR LDA, TF-IDF
class LDA_TFIDF(LDA) :
    def train_model(self, dataset, hyperparams=None, top_words=10):
        """
        Train the model and return output

        Parameters
        ----------
        dataset : dataset to use to build the model
        hyperparams : hyperparameters to build the model
        top_words : if greater than 0 returns the most significant words for each topic in the output
                 (Default True)
        Returns
        -------
        result : dictionary with up to 3 entries,
                 'topics', 'topic-word-matrix' and
                 'topic-document-matrix'
        """
        if hyperparams is None:
            hyperparams = {}

        if self.use_partitions:
            train_corpus, test_corpus = dataset.get_partitioned_corpus(use_validation=False)
        else:
            train_corpus = dataset.get_corpus()

        if self.id2word is None:
            self.id2word = corpora.Dictionary(dataset.get_corpus())

        if self.id_corpus is None:
            self.id_corpus = [self.id2word.doc2bow(document)
                              for document in train_corpus]
            self.tfidf = TfidfModel(self.id_corpus, normalize=True, smartirs='ntc')
            self.tfidfcorpus=self.tfidf[self.id_corpus]
            # self.tfidfcorpus=corpus2csc(self.tfidf[self.id_corpus])

        if "num_topics" not in hyperparams:
            hyperparams["num_topics"] = self.hyperparameters["num_topics"]
            

        # Allow alpha to be a float in case of symmetric alpha
        if "alpha" in hyperparams:
            if isinstance(hyperparams["alpha"], float):
                hyperparams["alpha"] = [
                    hyperparams["alpha"]
                ] * hyperparams["num_topics"]

        hyperparams["corpus"] = self.tfidfcorpus
        hyperparams["id2word"] = self.id2word
        self.hyperparameters.update(hyperparams)

        self.trained_model = ldamodel.LdaModel(**self.hyperparameters)

        result = {}

        result["topic-word-matrix"] = self.trained_model.get_topics()

        if top_words > 0:
            topics_output = []
            for topic in result["topic-word-matrix"]:
                top_k = np.argsort(topic)[-top_words:]
                top_k_words = list(reversed([self.id2word[i] for i in top_k]))
                topics_output.append(top_k_words)
            result["topics"] = topics_output

        result["topic-document-matrix"] = self._get_topic_document_matrix()

        if self.use_partitions:
            new_corpus = [self.id2word.doc2bow(
                document) for document in test_corpus]
            if self.update_with_test:
                self.trained_model.update(new_corpus)
                self.id_corpus.extend(new_corpus)

                result["test-topic-word-matrix"] = self.trained_model.get_topics()

                if top_words > 0:
                    topics_output = []
                    for topic in result["test-topic-word-matrix"]:
                        top_k = np.argsort(topic)[-top_words:]
                        top_k_words = list(
                            reversed([self.id2word[i] for i in top_k]))
                        topics_output.append(top_k_words)
                    result["test-topics"] = topics_output

                result["test-topic-document-matrix"] = self._get_topic_document_matrix()

            else:
                test_document_topic_matrix = []
                for document in new_corpus:
                    document_topics_tuples = self.trained_model[document]
                    document_topics = np.zeros(
                        self.hyperparameters["num_topics"])
                    for single_tuple in document_topics_tuples:
                        document_topics[single_tuple[0]] = single_tuple[1]

                    test_document_topic_matrix.append(document_topics)
                result["test-topic-document-matrix"] = np.array(
                    test_document_topic_matrix).transpose()
        return result