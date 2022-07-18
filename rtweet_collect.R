library(rtweet)


##1
###################################
## authenticate via access token ##
###################################

token <- create_token(
  app = "mobilise poland",
  consumer_key = "000000000000000000",
  consumer_secret = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  access_token = "AAAAAAAAAAAAAAAAAAAAA-000000000000000000000000000",
  access_secret = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")



##2
#####################
## Data Collection ##        --> Think about adding the updated date to not erase the previous collected tweets !!!
#####################

## Collecting Belarus hashtags 

#  !!! LAST RUN: 19/04/2022 !!!

BelTweets_1 <- search_tweets(
      "#Minska OR #LukashenkoOut OR #BelarusStrong OR #LukashenkoLeaveNow OR #лукашенкоубийца OR #БеларусьРеволюція", n = 18000000, include_rts = TRUE, retryonratelimit =TRUE  )
      BelTweets_1=BelTweets_1[order(BelTweets_1$created_at),]
  
BelTweets_2 <- search_tweets(
    "#ЛукашенкоУходи OR #LukashenkoGoAway OR #BelarusProtest OR #BelarusProtests OR #FreeBelarus OR #zhyvebelarus OR #ЖывеБеларусь OR 
    #prayforbelarus OR #helpbelarus OR #freebelarus OR #freedombelarus OR #savebelarus", n = 18000000, include_rts = TRUE, retryonratelimit =TRUE  )
    BelTweets_2=BelTweets_2[order(BelTweets_2$created_at),]
      
BelTweets_3 <- search_tweets(
      "#BelarusFreedom OR #Belarus2020 OR #Lukashenko OR #Belaruslivesmatter OR #BelarusSolidarity OR #FreedomforBelarus OR #ЖывэБеларусь OR #ЖивеБеларусь OR #Беларусь  OR 
      #Беларусь2020  OR #LongLiveBelarus  OR #Уходи  OR #автозак", n = 18000000, include_rts = TRUE, retryonratelimit =TRUE  )
    BelTweets_3=BelTweets_3[order(BelTweets_3$created_at),]     
  

BelTweets_7 <- search_tweets("#belarus OR #minsk", n = 1800000, include_rts = TRUE, retryonratelimit =TRUE)
    BelTweets_7=BelTweets_7[order(BelTweets_7$created_at),]  
  
BelTweets_8 <- search_tweets("#лукашенковавтозак OR #ЖывеБелорусь OR #жыве_беларусь OR #лошкипетушки OR #Белoрусь OR #Минск OR #ЖивеБілорусь #MinskMaidan", n = 1800000, include_rts = TRUE, retryonratelimit =TRUE)
    BelTweets_8=BelTweets_8[order(BelTweets_8$created_at),]
    
BelTweets_9 <- search_tweets("#belaruselections OR #belaruselections2020 OR #LukashenkoLeave OR #Minskprotests OR #standwithbelarus OR #BelarusRevolution OR #BelarusWatch OR #belarusstrike OR #ЛукашескуТыСледующий", n = 1800000, include_rts = TRUE, retryonratelimit =TRUE)
    BelTweets_9=BelTweets_9[order(BelTweets_9$created_at),]
    
BelTweets_10 <- search_tweets("#ЛукашенкоДиктатор OR #лукашенкокровавыйдиктатор OR #лукашенкокровавыйубийца OR #ЛукашенкоДиктатор OR #Лукашеску OR #кровавыйтаракан OR #СтопТаракан OR #сашатрипроцента", n = 1800000, include_rts = TRUE, retryonratelimit =TRUE)
    BelTweets_10=BelTweets_10[order(BelTweets_10$created_at),]
    
BelTweets_11 <- search_tweets("#she4belarus OR #беларуски_против_диктатуры OR #беларускі_супраць_дыктатуры", n = 1800000, include_rts = TRUE, retryonratelimit =TRUE)
    BelTweets_11=BelTweets_11[order(BelTweets_11$created_at),]
    

    
      
    
   ### Not be run (captured in BelTweets_7) ! 
# Only tweets posted from Belarus (radius is calculated from Tcherven in the center of the country)     
    #BelTweets_4 <- search_tweets("#belarus OR #minsk", n = 1800000, include_rts = TRUE, retryonratelimit =TRUE, geocode ="53.70,28.41,300mi")
    #BelTweets_4=BelTweets_4[order(BelTweets_4$created_at),]   
    
# Only tweets posted in Belorussian      
    #BelTweets_5 <- search_tweets("#belarus OR #minsk", n = 1800000, include_rts = TRUE, retryonratelimit =TRUE, lang = 'be')
    #BelTweets_5=BelTweets_5[order(BelTweets_5$created_at),] 
    
# Only tweets posted in Russian      
    #BelTweets_6 <- search_tweets("#belarus OR #minsk", n = 1800000, include_rts = TRUE, retryonratelimit =TRUE, lang = 'ru')
    #BelTweets_6=BelTweets_6[order(BelTweets_6$created_at),]

    
  
    ##3
    ################################
    ## Save and export data files ##       --> Think about adding the updated date to not erase the previous collected tweets !!!
    ################################
    
## Append all the csv files
BelTweets_Last <- rbind(BelTweets_1,BelTweets_2,BelTweets_3,BelTweets_7,
                        BelTweets_8,BelTweets_9,BelTweets_10,BelTweets_11)
    
## Find & Remove Duplicates (Tweets have an unique ID, if the tweet unique ID appears more than once, we keep only one line)
BelCleanNew <- BelTweets_Last[!duplicated(BelTweets_Last$status_id),]
BelCleanNew=BelCleanNew[order(BelCleanNew$created_at),]      
    
    
## Save the master file      --> Think about adding the updated date to not erase the previous collected tweets !!!
save_as_csv(BelCleanNew,"~/MOBILISE Data Collection/BelTweets_Master/BelTweets_Last_20220419.csv")
    

    ## THE END ##
    
    