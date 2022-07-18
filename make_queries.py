# Multiple queries to Twitter archive
import pandas as pd
import yaml
import datetime
from srch_v2 import make_queries_and_merge

###############################################################################
# Import authentification configuration
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
token = config['bearer_token']
###############################################################################
# Import queries that we generated
from belarus_queries import queries_to_run
# or make a custom query
# queries_to_run = (#Minska OR #LukashenkoOut OR #BelarusStrong OR #LukashenkoLeaveNow)

#################### TWARC WRAPPER ##################################

# Configure start and end time for the API request itself (don't put it inside the query string)
start_time = datetime.datetime(2020, 6, 1, 0, 0, 0, 0, datetime.timezone.utc)
end_time = datetime.datetime(2020, 6, 2, 0, 0, 0, 0, datetime.timezone.utc)

############# PERFORM SEARCH ##############
make_queries_and_merge(token, queries_to_run, 
                        start_time=start_time, end_time=end_time, max_results=100,
                        folder="data/newdata",
                        tweets_to_csv=True,
                        # Wite flattened tweets. Append to a single JSONL for all queries
                        pglimit=None,
                        save_groups=True)