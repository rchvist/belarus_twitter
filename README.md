# Twitter activity during Protests and mobilization in Belarus (2020)
## Description
This repo contains code used for the collection, descriptive quantitative analysis, and topic modelling of Twitter data related to the crisis of 2020 in Belarus. This was done as part of the [MOBILISE research project](https://mobiliseproject.com/). Each step of our analysis is extensively described [here](docs/analysis_steps.pdf).
The main goal of this repo is to allow researchers to replicate our methodology, as well as re-purpose code for future projects.


##	1/ Collection of tweets

  *Note : It will now be faster to collect tweets using the official [Tweet Downloader](https://developer.twitter.com/apitools/downloader). Input queries (see `belarus_queries.py`) and direct download CSV files. Make sure to check all fields and extensions except polls.* 
<details>
    <summary>Click to expand</summary>

###	Original batch
Early data was collected using the R plugin [RTweet](https://github.com/ropensci/rtweet) which requires legacy API V1 credentials from Twitter.  
* `rtweet_collect.R` was executed weekly. Input your own credentials.

### Second batch
Additional data was collected using Twitter API V2 [Academic access](https://developer.twitter.com/en/products/twitter-api/academic-research) and the Python library [Twarc](https://github.com/docnow/twarc). We cover missing dates and performed keyword augmentation (new hashtags). 
* `pip install twarc==2.9.5`
* `make_queries.py` to execute a batch of requests and get 1 CSV per query group.
* `new_hashtag_list` contains hashtags and keywords that were used to extend the dataset

### Converting JSON (V2) to (V1) CSV
We use the Twarc_CSV plugin to flatten JSON data into a format we can merge with the CSV produced by Rtweet. 
Requires the separate plugin [twarc-csv](https://github.com/DocNow/twarc-csv) :  
* `pip install twarc-csv==0.5.2`
* `csv_conversion.ipynb` this notebook can be used to replicate this step and get .csv from JSON 

### Count of tweets
When we only need the *count of tweets* for a certain request (timeframe, language), we only make a [Count request](https://developer.twitter.com/en/docs/twitter-api/tweets/counts/introduction).  
This is demonstrated in `counts.ipynb`.

### Dependencies :
 - `srch_v2.py` custom wrapper over Twarc with search functions
 - `belarus_queries.py` contains hashtags we followed
 - `config.yaml` for credentials
 - `rtweet_conversion.py` contains a data translation table + specific fields from API V2 to Rtweet format (API V1)
 - `v2_csv_converter.py` custom wrapper over twarc_csv to extract specific data, flatten JSON into CSV 

Example config.yaml :
```yaml
api_key: 0000000000000000000000000
api_secret: 00000000000000000000000000000000000000000000000000
bearer_token : 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
access_token : 00000000000000000000-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
access_token_secret : 000000000000000000000000000000000000000000000
sql_debug: false
rapidapi_key: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
```
</details>  

## 2/ Data cleaning and exploration

Exploratory data analysis (EDA) and cleaning is performed in the included notebook `beltweets_analysis.ipynb`

Included code : 
* Merging data from API V1 and API V2
* Data cleaning
* Input of missing languages 
* Visualisations for statistical analysis, tweets over time, users, top RT...
* Export of a subset for NLP (tweets in english and russian)
* Other early EDA steps that we abandonned (bot detection models...)

Dependencies :
- `rtweet_conversion.py` contains a data translation table + specific fields from API V2 to Rtweet format (API V1)
- `dtypes.py` contains the data scheme used when importing .csv with Pandas


## 3/ Topic modelling and other NLP analyses
### Pre-processing
`nlp_preprocessing.py` contains most text cleaning functions. The main func `preprocess()` will be used before every topic model, with different params such as maximum document frequency, n-grams... See the [doc](docs/analysis_steps.pdf) for an extensive description of this pipeline. Lemmatization, which relies on [Spacy](https://spacy.io/), is responsible for most of the execution time.  

*Note : When planning for many experiments, `nlp_spacy_parser.py` was meant to be used to parse the entire dataset once and for all with Spacy (POS tagging, lemmatization, named entity recognition), and save the result either in an array column of the .csv, or separate .spacy files. We can alter the `preprocess()` function to use read_spacy_from_arrays() and load a [Doc()](https://spacy.io/api/doc) item for each tweet, and the doc.lemma_ item, instead of performing lemmatization each time.*

### Hyperparameter search
For CTM, hyperparameters were chosen using the library [OCTIS](https://github.com/mind-Lab/octis).  
For instance, parameters used in `TM_run_CTM.py` : dropout = 0.09891608522984201, num_neurons=300  
The script `TM_run_OCTIS_optimization.py` provides an example of an early hyperparameter search.

### Topic models
These scripts are used to run batches of models for multiple subsets of data (per day/month, and per language), and multiple values of K topics. 
They include automatic selection of "best" K, output topics and PyLDAVis visualisations.  Links to the required libraries are included.   
* [Biterm Topic Model](https://pypi.org/project/bitermplus/) (BTM) : see `TM_run_BTM.py`
* CTM ([Octis version](https://github.com/mind-Lab/octis)) for daily data : run `TM_run_CTM_octis.py`  
* CTM ([Origin version](https://github.com/MilaNLProc/contextualized-topic-models)) for daily data : run `TM_run_CTM_vanilla.py`  
* Mixture models (Octis implementation) : see `TM_run_MM.py` 

### Interpretation and visualization of topics
Comparison of performance (coherence, diversity) is visualized in included notebooks `TM_CTM_octis_viz.ipynb`,  `TM_BTM_viz.ipynb`  
They also contain exploratory analysis of the topics, such as plotting topic dynamics.

### Sentiment analysis
See `nlp_sentdetect.py`

