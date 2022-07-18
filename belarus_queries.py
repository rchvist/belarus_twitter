# 4 groupes de requêtes générées en amont (augmentation du dataset)
# 1. Une chaîne de caractères, les hashtags d'origine
# 2,3 et 4 : listes de chaînes de caractères (#nouveau_hashtag - (exclusion de la requête d'origine))
#   2. 2nde vague de hashtags
#   3. 2nde vague de hashtags, alphabet cyrillique uniquement
#   4. 3e vague de hashtags et mots-clé repérés à partir de "trending en biélorussie"
queries_to_run=[
### 1
'''(#Minska OR #LukashenkoOut OR #BelarusStrong OR #LukashenkoLeaveNow OR #лукашенкоубийца OR #БеларусьРеволюція OR #belarus OR #minsk OR #ЛукашенкоУходи OR 
#LukashenkoGoAway OR #BelarusProtest OR #BelarusProtests OR #FreeBelarus OR #zhyvebelarus OR #ЖывеБеларусь OR #prayforbelarus OR #helpbelarus OR #freebelarus
 OR #freedombelarus OR #savebelarus OR #BelarusFreedom OR #Belarus2020 OR #Lukashenko OR #Belaruslivesmatter OR #BelarusSolidarity OR #FreedomforBelarus OR 
 ЖывэБеларусь OR #ЖивеБеларусь OR #Беларусь OR #Беларусь2020 OR #LongLiveBelarus OR #Уходи OR #ЖывеБелорусь OR #жыве_беларусь OR #Минск OR #ЖивеБілорусь #MinskMaidan
OR #belaruselections OR #belaruselections2020 OR #LukashenkoLeave OR #Minskprotests OR #standwithbelarus OR #BelarusRevolution OR #BelarusWatch OR #belarusstrike OR
#ЛукашескуТыСледующий OR #ЛукашенкоДиктатор OR #лукашенкокровавыйдиктатор OR #лукашенкокровавыйубийца OR #ЛукашенкоДиктатор OR #Лукашеску OR #кровавыйтаракан OR #СтопТаракан)''',
### 2
['''(#lukaschenko OR #tsikhanouskaya OR #bélarus OR #bielorussia OR #bielorrusia OR #belaruspresidentialelection OR #lukashenka OR #tikhanovskaya OR #belarusian OR #bielorussie
OR #лукашенко OR #brest OR #belaruselection OR #solidaritywithbelarus OR #protestsinbelarus OR #bahinskaya OR #belarusians OR #białoruś OR #loukachenko OR #hrodna OR #belarús 
OR #tichanowskaja OR #grodno OR #biélorussie OR #kalesnikava OR #highlightbelarus OR #belarusinaction OR #bialorus OR #mińsk OR #belaruselection OR #лукашенко OR #kolesnikova 
OR #romanbondarenko OR #белоруссия OR #strikebelarus OR #okrestina OR #belsat OR #baltkrievija OR #mariakolesnikova OR #weißrussland OR #kolesnikowa OR #svetlanatikhanovskaya
OR #тихановская OR #omon OR #ninabaginskaya OR #supportbelarus OR #women_in_white OR #weissrussland OR #wolnabiałoruś)''',

'''(#alexanderlukashenko OR #хабаровск OR #homiel OR #bandarenka OR #alexievich OR #тихоновоская OR #минске OR #białoruś OR #kyiv OR #salihorsk OR #womenrisingfordemocracy
OR #orlovskaya OR #belarusnow OR #białoruśprotesty OR #withbelarus OR #belarusianlivesmatter OR #тихановская OR #svetlanasugako OR #вольнаябеларусь OR #bielarus
OR #mogilev OR #nadezhdabrodskaya OR #ramanbandarenka OR #білорусь)'''],
### 3

['''(#живэбеларусь OR #живебелорусь OR #протесты OR #омон OR #жывэбелорусь OR #выборы2020 OR #беларуси OR #противлукашенко OR #беларусьмысвами OR #живэбелорусь OR #мглу 
OR #сябрысила OR #каратель OR #установлен OR #лукашэнка OR #свободнаябеларусь OR #жывeбеларусь OR #брест OR #выборыбелоруссия OR #выборыбеларусь2020 OR #светланатихановская 
OR #протестыбеларусь OR #бчб OR #они_давили_народ OR #длятрибунала OR #новополоцк OR #протестыбелоруссия OR #верымможампераможам OR #беларус OR #мінск OR #цепкало OR #минска
 OR #чтобыпомнили OR #ягуляю OR #явыхожу OR #вясна OR #беларусьвыборы OR #беларусі OR #вов OR #мвд OR #оон OR #гродно OR #ямы97 OR #беларусов OR #менск OR #забастовка OR
#михнович_кирилл OR #странадляжизни OR #нехта OR #полоцк OR #новости_беларуси OR #рб OR #гомель OR #выбары2020 OR #таракануходи OR #шишмаков OR #ціханоўская OR #змагары
OR #верамможампераможам OR #свободасобраний OR #витебск OR #живибеларусь OR #бнф OR #произвол OR #пытки OR #ямыбатька OR #мінськ OR #белоруссии)''',

'''(#игорькоротченко OR #змагар OR #белорусь OR #плошча OR #колесникова OR #белаз OR #литва OR #россияпротивлукашенко OR #могилев OR #язалукашенко OR #протестывбеларуси 
OR #беларуссия OR #лукошенкоуходи OR #борисьбеларусь OR #божахранібеларусь OR #бабарико OR #беларусьживи OR #свободумарииколесниковой OR #білоруськіпротести OR #кобрин 
OR #уняволі OR #екрп OR #беларуских OR #безправанарасправу OR #куштауживи)'''],

### 4
'''(psiphon OR Корж OR Коржа OR Жодино OR #BelarusSolidarity OR Belarusian OR #выборы OR молотова OR СИЗО OR МЗКТ OR #FreeBelarus OR 
#Belarus2020 OR Латушко OR дзякуй OR Окрестина OR Окрестино OR Реальный Брест OR Пушкинской OR #belarussolidarity OR цепкало)'''
]
