# dict of {V2 field name:Rtweet name} : columns that we get from V2 and we can rename as is
torename = {'id': 'status_id',
'referenced_tweets.replied_to.id': 'reply_to_status_id',
'referenced_tweets.retweeted.id': 'retweet_status_id',
'in_reply_to_user_id': 'reply_to_user_id',
'referenced_tweets.quoted.id': 'quoted_status_id',
'author_id': 'user_id',
'retweeted_user_id': 'retweet_user_id',
'quoted_user_id': 'quoted_user_id',
'created_at': 'created_at',
'text': 'text',
'lang': 'lang',
'source': 'source',
'public_metrics.like_count': 'favourites_count',
'public_metrics.quote_count': 'quote_count',
'public_metrics.reply_count': 'reply_count',
'public_metrics.retweet_count': 'retweet_count',
'author.created_at': 'account_created_at',
'author.username': 'screen_name',
'author.name': 'name',
'author.description': 'description',
'author.entities.url.urls': 'profile_expanded_url',
'author.location': 'location',
'author.pinned_tweet_id': 'status_url',
'author.profile_image_url': 'profile_image_url',
'author.protected': 'protected',
'author.public_metrics.followers_count': 'followers_count',
'author.public_metrics.following_count': 'friends_count',
'author.public_metrics.listed_count': 'listed_count',
'author.public_metrics.tweet_count': 'statuses_count',
'author.url':'url',
'author.verified': 'verified',
 # Geo fields in V1 : The 'place' object is always present when a Tweet is geo-tagged,
 # while the 'coordinates' object is only present (non-null) 
 # when the Tweet is assigned an exact location.
'geo.coordinates.coordinates': 'coords_coords',
'geo.country': 'country',
'geo.country_code': 'country_code',
'geo.full_name': 'place_full_name',
'geo.geo.bbox': 'bbox_coords',
'geo.name': 'place_name',
'geo.place_id': 'place_url',
'geo.place_type': 'place_type',
'entities.cashtags' : 'symbols',
}
# Columns from rtweet that we ended up dropping after analysis
dropcols=['display_text_width',
    'urls_url', # Normalement c'est la liste des URLS mentionnées, tronqué par rtweets. urls_expanded_url contient la 1ère (?) de ces URLS
    'urls_t.co', 
    'geo_coords', # Vide dans Rtweets, on conserve "coords_coords"
    'favorite_count', # Même chose que les likes, existe en double dans rtweets
    'profile_banner_url', 
    'profile_background_url',
    'media_expanded_url', # C'est le lien vers l'image dans le status, on peut le reconstruire si besoin. Dans la v2 pas d'expanded_url.
    'media_t.co',
    # Ext_media : Dans la v1, renvoie une liste si plusieurs medias (4 photos max)
    # On conserve ext_media_url
    'ext_media_expanded_url', 
    'ext_media_t.co',
    'profile_url', # N'existe pas V2
    'profile_expanded_url',
    'account_lang', # Vide
    'reply_to_screen_name', # Beaucoup de données manquantes, TODO les ré-imputer depuis le JSON ?
    'profile_image_url', # Inutile
    'media_url', # Redondant avec ext_media_url
    # many retweet fields are'nt useful since they inherit everything from the original tweet
    'retweet_location', # ces deux là il faut les remettre
    'retweet_description', # ces deux là il faut les remettre, le retwitteur diffère du twitteur
    'retweet_verified', 
    'retweet_text',
    'retweet_source',
    'retweet_retweet_count',
    'retweet_favorite_count']
toexpand={
 'in_reply_to_user_id': '', 
 'referenced_tweets.id': '', # expansion field
 'referenced_tweets.id.author_id': '',
 'entities.mentions' : '', # V1 mentions_user_id : tweet["entities']['mentions']['id'], # dict des mentions
 'attachments.media_keys': '', # expansion field
 'attachments.media': '', # dict des medias
 'geo.place_id': '', # C'est un champ clé d'expansion
 'entities.urls': ''
  }


todrop_V2 = ['attachments.poll_ids', 
'attachments.poll.duration_minutes',
'attachments.poll.end_datetime',
'attachments.poll.id',
'attachments.poll.options',
'attachments.poll.voting_status',
'attachments.poll_ids',
'conversation_id',
'withheld.scope',
'withheld.copyright',
'withheld.country_codes',
'reply_settings',
'possibly_sensitive',
'context_annotations',
'author.entities.description.cashtags',
'author.entities.description.hashtags',
'author.entities.description.mentions',
'author.entities.description.urls',
'author.withheld.scope',
'author.withheld.copyright',
'author.withheld.country_codes',
'geo.geo.type', # V1.1 place.id.bounding_box.type, n'est pas dans Rtweet
'geo.id', # Pas dans Rtweet
'__twarc.retrieved_at',
'__twarc.url',
'__twarc.version']

to_ignore_V1 = ['display_text_width',
'urls_url', # Normalement c'est la liste des URLS mentionnées, tronqué par rtweets. urls_expanded_url contient la 1ère (?) de ces URLS
'urls_t.co', 
'geo_coords', # Vide dans Rtweets, on conserve "coords_coords"
'favorite_count', # Même chose que les likes, existe en double dans rtweets
'profile_banner_url', 
'profile_background_url',
'media_expanded_url', # C'est le lien vers l'image dans le status, on peut le reconstruire si besoin. Dans la v2 pas d'expanded_url.
'media_t.co',
# Ext_media : Dans la v1, renvoie une liste si plusieurs medias (4 photos max)
'ext_media_expanded_url', 
'ext_media_type', 
'ext_media_url', 
'ext_media_t.co',
'profile_url', # N'existe pas V2
'account_lang'] # Deprecated, utiliser le langage du pinned tweet

# Rtweet fields that we will extract and create when converting to CSV
extra_cols =['is_retweet',
 'is_quote',
 'retweet_location',
 'quoted_text',
 'retweet_description',
 'quoted_screen_name',
 'retweet_source',
 'quoted_name',
 'retweet_statuses_count',
 'quoted_verified',
 'retweet_favorite_count',
 'retweet_verified',
 'quoted_created_at',
 'quoted_description',
 'retweet_screen_name',
 'quoted_friends_count',
 'retweet_friends_count',
 'quoted_location',
 'retweet_followers_count',
#  'media_url',
 'media', ################## PROBLEM 03/06
 #### APRES AVOIR RECUP LES TWEETS SUR LE DOWNLOADER DE TWITTER
 # Cette colonne apparaissait en extra_cols du Converter_L 
 'ext_media_url',
 'ext_media_type',
 'quoted_retweet_count',
 'quoted_source',
 'quoted_followers_count',
#  'media_type',
 'retweet_retweet_count',
 'retweet_created_at',
 'quoted_statuses_count',
 'quoted_favorite_count',
 'mentions_user_id',
 'retweet_text',
 'retweet_name',
 'mentions_screen_name',
 "hashtags",
 'urls_expanded_url',
 'ext_media_views' # ajouté dans le converter v2
 ]

dropcols_v1=['display_text_width',
'urls_url', # Normalement c'est la liste des URLS mentionnées, tronqué par rtweets. urls_expanded_url contient la 1ère (?) de ces URLS
'urls_t.co', 
'geo_coords', # Vide dans Rtweets, on conserve "coords_coords"
'favorite_count', # Même chose que les likes, existe en double dans rtweets
'profile_banner_url', 
'profile_background_url',
'media_expanded_url', # C'est le lien vers l'image dans le status, on peut le reconstruire si besoin. Dans la v2 pas d'expanded_url.
'media_t.co',
# Ext_media : Dans la v1, renvoie une liste si plusieurs medias (4 photos max)
# On conserve ext_media_url
'ext_media_expanded_url', 
'ext_media_t.co',
'profile_url', # N'existe pas V2
'profile_expanded_url',
'account_lang', # Vide
'reply_to_screen_name', # Beaucoup de données manquantes, TODO les ré-imputer depuis le JSON ?
'profile_image_url', # Inutile
'media_url' # Redondant avec ext_media_url
]