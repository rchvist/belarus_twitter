import json
from jsonmerge import Merger
import os
from twarc import ensure_flattened
from twarc.client2 import Twarc2
from twarc_csv import CSVConverter
import datetime
from twarc.expansions import (
    EXPANSIONS,
    TWEET_FIELDS,
    USER_FIELDS,
    MEDIA_FIELDS,
    POLL_FIELDS,
    PLACE_FIELDS,
    ensure_flattened,
)

# CSV Conversion
from v2_csv_converter import  DataFrameConverter,  torename, extra_cols, dropcols
  # Columns that we want to keep for now
  # dropcols = rtweet names for columns that we ended up deleting
  # torename = match rtweet names with V2 names
  # extra_cols = some interesting fields that didn't exist in V1 and we wanted to collect for V2
tokeep=[k for k,v in torename.items() if v not in dropcols]+extra_cols+['context_annotations','entities.annotations', 'geo.id']

# Schema to merge several paginated responses into one
schema_no_dupes = {
  "properties": {
    "data": {
      "type":'array',
      "mergeStrategy": "arrayMergeById",
      'items' : {
        "mergeStrategy":"overwrite"
      }
    },
        "includes": {
      "type": "object",
      "properties": {
        "users": {
          "mergeStrategy": "append"
        },
        "tweets": {
          "mergeStrategy": "append"
        },
        "polls": {
          "mergeStrategy": "append"
        },
        "places": {
          "mergeStrategy": "append"
        },
        "media": {
          "mergeStrategy": "append"
        }
      }
    },
    "meta": {
      "type": "object",
      "properties" : {
        "newest_id": {
          "mergeStrategy" : "discard",
          "mergeOptions": {"keepIfUndef" : 'true'}
        },
        'oldest_id': {
          "mergeStrategy" : "overwrite"
        },
        'result_count': {
          "mergeStrategy" : "discard",
          "mergeOptions": {"keepIfUndef" : 'true'}
        }       
      }
    }
  }
}


merger = Merger(schema_no_dupes)

def merge_paginated_query(token=None,
                        query=None, 
                        start_time=None, 
                        end_time=None, 
                        max_results=100,
                        pglimit=None,
                        pages_to_file=None,
                        tweets_to_file=None, 
                        write_method='a') :

  # Your bearer token here (replace "token")
  t = Twarc2(bearer_token=token)


  tokeep=[k for k,v in torename.items() if v not in dropcols]+extra_cols+['context_annotations','entities.annotations', 'geo.id']


  search_results = t.search_all(query=query, start_time=start_time, end_time=end_time, max_results=max_results,
                          # # Specify expansions and fields. Tweets will include keys for those expansions (e.g. user.id) and all tweet_fields. 
                          # # Objects (e.g. users) will be collected
                          # # Default all expansions, all fields
                          # expansions=','.join([e for e in EXPANSIONS if 'poll' not in e]),
                          # poll_fields=None,
                          # user_fields=TWEET_FIELDS, media_fields=, poll_fields=, place_fields=
                          # tweet_fields=tweetfields
                          )
  p=0
  query_dataset={}     
  for page in search_results:
    write_method = 'a'
    if pages_to_file :
        # Dump page as a new line in a JSONL file, either append (a) or new (w)
        with open(pages_to_file, write_method) as f:
            f.write(json.dumps(page) + "\n")

    # Merge dict query_dataset with previous pages
    query_dataset = merger.merge(query_dataset, page)

    # print("Wrote page "+str(p+1)+" of results with "+str(len(page["data"]))+" tweets pulled...", end='\r')
    # Print page

    if p!=0 and page and 'meta' in page :
        # Unless first page or empty page, increment the number of results in meta object
        query_dataset['meta']['result_count']+=page['meta']['result_count']
        # print('Query results count : %d\r'%query_dataset['meta']['result_count'], end="", flush=True)

    elif not page :
      print('\nNo more results for this query')


    # pagelist.append(page)
    # # Append the page to a list

    for tweet in ensure_flattened(page):
        # Do something with the "flattened" tweet (each tweet contains all expansions)
        if tweets_to_file :
          # Append tweets inline to file
            with open(tweets_to_file, write_method) as f:
                f.write(json.dumps(tweet) + '\n')

    if pglimit and p==pglimit-1 :
        # Limit number of collected pages
        break
    p+=1
  return(query_dataset)

############# MULTIPLE SEARCHES ##############

# merge_paginated_query calls a generator, max_results is max tweets per page, not total, 100 is max when using all expansions.
def make_queries_and_merge(token, queries, start_time, end_time, max_results=100,
                                        folder="",
                                        tweets_to_csv=False,
                                        pglimit=None,
                                        save_groups=True, split_time=None) :
  """
    A simple iterator for the "Search All" twarc function to perform several requests on the Twitter archive,
    deal with pagination, and merge all responses into one, excluding duplicate tweets. 
    Can output a json, jsonl and/or CSV for each query group.

    Args:
        queries (str|list):
            Either a single query string (< 1024 characters), list of strings, or list of lists.
            If list, will output a json and/or csv file for each query.
            If list of lists, the inner lists are treated as grouped queries. Will output a json and/or csv file for each group.
        start_time (datetime):
            Return all tweets after this time (UTC datetime). If none of start_time, since_id, or until_id
            are specified, this defaults to 2006-3-21 to search the entire history of Twitter.
        end_time (datetime):
            Return all tweets before this time (UTC datetime).
        max_results (int):
            The maximum number of results per request. Max is 500.
        save_groups (bool): default True, whether to output a separate file for each element of the list, 
        potentially keeping duplicates inbetween query groups
        folder: where to output the resulting json and/or csv
        pglimit: hard limit on the number of pages to request (for testing)
        search_results: paginated response from twarc.search_all()
        tweets_to_csv: Export json files to custom CSV format
        split_time (int or None) : 
          If int, returns search results for slices of (split_time) days between start_time and end_time.
          Last time period will be truncated in case of uneven slices.
          default=None returns the entire time period

    Returns:
        dict: Pages merged into a search response format (data, includes, meta, __twarc)  from all pages
    """
  start_period=start_time
  
  os.makedirs(folder, exist_ok=True)
  
  while start_period < end_time :
    if split_time is None :
      delta = end_time-start_time
    else :
      delta = datetime.timedelta(split_time)
    end_period=start_period+delta
    
    if end_period>end_time :
        end_period=end_time
    

  
    if isinstance(queries, str) :
      queries=[[queries]]
    dataset = {}
    # per_query_data=[]
    # if save_groups and len(queries)>1 :
      # per_group_data=[]
    for g, group in enumerate(queries) :
      group_dataset={}
      if isinstance (group, str) :
        group = [group]
      for q, query in enumerate(group) :
          print(f"\nSearching tweets from group " +str(g+1)+ ", query "+str(q+1)+ f" : \"{query[:50]}...\" tweets from {start_period} to {end_period}...", flush=True)
          
          # Merge paginated responses into one, increment a file with tweets
          query_dataset = merge_paginated_query(token=token, query=query, start_time=start_period, end_time=end_period, max_results=max_results, 
                                              
                                              pglimit=pglimit)

          group_dataset=merger.merge(group_dataset, query_dataset)
          
          # Merge entire dataset with this query
          dataset = merger.merge(dataset, query_dataset)

          # per_query_data.append(query_dataset)

          if q!=0 and query_dataset and 'meta' in query_dataset :
              # Unless first query or empty query, increment the number of results in meta object
              group_dataset['meta']['result_count']+=query_dataset['meta']['result_count']
              dataset['meta']['result_count']+=query_dataset['meta']['result_count']
              print('\nGroup ' + str(g+1) + ', Query ' + str(q+1) + ' results count : '+ str(query_dataset['meta']['result_count']) +". Total dataset = "+ str(len(dataset['data'])) + ' tweets', flush=True)
          elif query_dataset and 'meta' in query_dataset :
              print('\nGroup ' + str(g+1) + ', Query ' + str(q+1) + ' results count : '+ str(query_dataset['meta']['result_count']) +". Total dataset = "+ str(len(dataset['data'])) + ' tweets', flush=True)
          if not query_dataset :
              print ('\nNo results for this query\n', flush=True)

      # per_group_data.append(group_dataset)
      if save_groups and len(queries)>1 :
        f_path= folder+f"/{start_period.strftime('%Y%m%d')}_{end_period.strftime('%m%d')}_query_gr_" + str(g+1)
        with open(f_path + '_response.json', "w+") as f:
          
            # Write file for the merged response for all queries
            f.write(json.dumps(group_dataset))
      
        
        if tweets_to_csv and save_groups and len(queries)>1 :
          # Convert group json to CSV
          print('Exporting CSV...')
          f= f_path + '_response.json'
          converter_L = DataFrameConverter(
          input_data_type="tweets",
          json_encode_all=False,
          json_encode_text=False,
          json_encode_lists=False,
          inline_referenced_tweets=False,
          merge_retweets=True,
          allow_duplicates=False,
          extra_input_columns=','.join(extra_cols),
          output_columns=','.join(tokeep)
          )
          with open(f, "r") as infile:
              outpath = f_path + '_dataset.csv'
              with open(outpath, "w") as outfile:
                  converter = CSVConverter(infile, outfile, converter=converter_L)
                  converter.process()

    # Export total dataset to json
    f_path= folder+f"/{start_period.strftime('%Y%m%d')}_{end_period.strftime('%m%d')}"
    with open(f_path+ '_response.json', 'w+') as f :
        f.write(json.dumps(dataset))
    # Convert total dataset to CSV
    if tweets_to_csv  :
      print('Exporting CSV for all queries...')
      f= f_path+ '_response.json'
      converter_L = DataFrameConverter(
      input_data_type="tweets",
      json_encode_all=False,
      json_encode_text=False,
      json_encode_lists=False,
      inline_referenced_tweets=False,
      merge_retweets=True,
      allow_duplicates=False,
      extra_input_columns=','.join(extra_cols),
      output_columns=','.join(tokeep)
      )
      with open(f, "r") as infile:
          outpath = f_path+ "_dataset.csv"
          with open(outpath, "w") as outfile:
              converter = CSVConverter(infile, outfile, converter=converter_L)
              converter.process()


    print(f"\nFinished period {start_period.strftime('%Y %m %d')} to {end_period.strftime('%Y %m %d')}", flush=True)
    # return(dataset, per_group_data, per_query_data)
    start_period=start_period+delta

  
  print("\nFinished.", flush=True)
  # return(dataset, per_group_data, per_query_data)