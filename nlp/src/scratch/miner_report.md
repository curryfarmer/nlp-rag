# Corpus Pattern Miner Report

Docs analysed: **296**


## money

| name | total_matches | docs_with_match | docs_with_match_pct | matches_per_doc | samples |
|---|---|---|---|---|---|
| strict_credits | 187 | 88 | 0.297 | 2.12 | 14.7 billion Phi Credits; 890 million Credits; 120 million Phi Credits |
| verbose_credits | 187 | 88 | 0.297 | 2.12 | 14.7 billion Phi Credits; 890 million Credits; 120 million Phi Credits |
| dollar_amount | 0 | 0 | 0.0 | 0.0 |  |
| financial_penalty_label | 52 | 37 | 0.125 | 1.41 | revenue of Φ 890,000,000 is consistent with Bureau projections issued at the ... |


## percent

| name | total_matches | docs_with_match | docs_with_match_pct | matches_per_doc | samples |
|---|---|---|---|---|---|
| digit_pct | 566 | 129 | 0.436 | 4.39 | 31%; 22%; 67% |
| digit_word_percent | 604 | 137 | 0.463 | 4.41 | approximately 31%; 31%; approximately 22% |
| share_of | 3 | 3 | 0.01 | 1.0 | 18% share; 3 share |


## date_pce

| name | total_matches | docs_with_match | docs_with_match_pct | matches_per_doc | samples |
|---|---|---|---|---|---|
| iso_pce | 1071 | 291 | 0.983 | 3.68 | 74-03-12 PCE; 28-09-14; 73-06-20 |
| quarter_pce | 92 | 29 | 0.098 | 3.17 | Q4 76 PCE; Q3 76 PCE; Q2 76 PCE |
| year_pce | 1300 | 212 | 0.716 | 6.13 | 12 PCE; 28 PCE; 27 PCE |
| year_ce | 274 | 68 | 0.23 | 4.03 | 2009 CE; 2115 CE; 2071 CE |
| year_dual | 117 | 36 | 0.122 | 3.25 | 03 PCE (2045 CE); 49 PCE (2091 CE); 52 PCE (2094 CE) |
| hour_time | 152 | 30 | 0.101 | 5.07 | 00:02; 00:07; 06:00 |


## codename

| name | total_matches | docs_with_match | docs_with_match_pct | matches_per_doc | samples |
|---|---|---|---|---|---|
| all_caps_word | 6970 | 281 | 0.949 | 24.8 | OPEN; CLAIROS; GOVERNANCE |
| designation | 906 | 179 | 0.605 | 5.06 | CGC-28-070; DOSSIER-0003; RED-29-02-08 |
| title_case_list | 502 | 218 | 0.736 | 2.3 | Founding, Mandate, and Market; Second, Wampa; Month, Year |
| seastitch_like | 5458 | 262 | 0.885 | 20.83 | CLAIROS; GOVERNANCE; CYANITE |
| quoted_string | 259 | 103 | 0.348 | 2.51 | the devastation caused; collective security determination; the Council |


## count

| name | total_matches | docs_with_match | docs_with_match_pct | matches_per_doc | samples |
|---|---|---|---|---|---|
| plain_int | 15228 | 296 | 1.0 | 51.45 | 74; 03; 28 |
| number_word | 1634 | 276 | 0.932 | 5.92 | one; twenty; four |
| approximately_n | 13738 | 296 | 1.0 | 46.41 | 74; 03; 28 |
| every_n_period | 7 | 4 | 0.014 | 1.75 | every 72 hours; every 8 minutes; every 22 minutes |


## entity

| name | total_matches | docs_with_match | docs_with_match_pct | matches_per_doc | samples |
|---|---|---|---|---|---|
| title_phrase | 10742 | 296 | 1.0 | 36.29 | The Dissolution of Wampa Robotics; Structural Analysis of Coordinated Corpora... |
| doctoral_credential | 7 | 6 | 0.02 | 1.17 | doctorate in autonomous systems design from an instit; masters does not becom... |


## structural

| name | total_matches | docs_with_match | docs_with_match_pct | matches_per_doc | samples |
|---|---|---|---|---|---|
| md_header | 2360 | 276 | 0.932 | 8.55 | # The Dissolution of Wampa Robotics: A Structural Analysis of Coordinated Cor... |
| md_bold_label | 1249 | 205 | 0.693 | 6.09 | **CLASSIFICATION:**; **DATE OF PASSAGE:**; **DOCUMENT CLASS:** |
| caps_label | 27 | 11 | 0.037 | 2.45 | CLASSIFICATION:; DISTRIBUTION:; TRANSMITTED: |
| colon_line | 303 | 131 | 0.443 | 2.31 | Classification:; Of particular operational relevance:; To be direct: |
| bullet | 343 | 62 | 0.209 | 5.53 | - ;  -  |
| classification | 133 | 107 | 0.361 | 1.24 | Classification: OPEN; Classification: RESTRICTED; CLASSIFICATION: CONFIDENTIAL |


## Top capitalised entity phrases (top 60)

```
   1234  PCE
   1146  CGC
    632  Haven
    381  TEC
    355  CE
    352  Edge
    322  Genesis Labs
    295  Section
    292  Cyanite Industries
    262  Bloc
    259  SECTION
    217  Cyanite
    214  ONE
    211  CYPHER
    204  Classification
    198  Conclave
    178  Phi Credits
    169  ONE Network Enterprises
    147  We
    138  Resolution
    135  Floodwall
    133  Renhwa Media
    123  Haven Island
    121  Phyrexis Group
    118  Clairos Governance Council
    115  Council
    114  Zonnon
    112  Distribution
    111  CGC Secretariat
    108  CLASSIFICATION
    103  Central Haven
    102  Clairos
     93  Cascade
     92  Kashikari
     92  Tier
     91  Haven Island Special Economic
     90  These
     87  CONFIDENTIAL
     87  Zone
     87  Date
     85  Phyrexis
     84  Phase
     83  CI
     79  Renhwa
     79  Director
     78  Sharpsea Bloc
     77  OPEN
     76  Article
     75  Document Reference
     74  The Edge Corporation
     74  Edge Research
     71  East Haven
     71  GL
     68  Sarento
     67  Somatic Augmentation
     65  Full
     64  Wampa
     63  RESTRICTED
     62  West Haven
     62  Division
```

## Top content bigrams (df 5..150)
```
  tf=  487 df= 72  genesis labs
  tf=  409 df=126  cyanite industries
  tf=  236 df= 99  haven island
  tf=  231 df= 88  one network
  tf=  221 df= 87  network enterprises
  tf=  207 df=116  clairos governance
  tf=  204 df=113  governance council
  tf=  183 df= 81  phi credits
  tf=  160 df= 53  central haven
  tf=  157 df= 55  renhwa media
  tf=  148 df= 69  phyrexis group
  tf=  126 df= 63  cgc member
  tf=  124 df= 62  cgc secretariat
  tf=  119 df= 44  east haven
  tf=  102 df= 53  edge corporation
  tf=  100 df= 44  north haven
  tf=   99 df= 66  special economic
  tf=   99 df= 66  economic zone
  tf=   99 df= 37  west haven
  tf=   98 df= 39  sharpsea bloc
  tf=   97 df= 41  member corporations
  tf=   97 df= 65  island special
  tf=   92 df= 28  somatic clinic
  tf=   91 df= 40  republic zonnon
  tf=   89 df= 27  somatic augmentation
  tf=   85 df= 81  document reference
  tf=   83 df= 21  edge research
  tf=   75 df= 41  cgc resolution
  tf=   75 df= 28  south haven
  tf=   69 df= 64  date pce
  tf=   69 df= 17  security directorate
  tf=   68 df= 15  biological growth
  tf=   67 df= 14  fiscal year
  tf=   67 df= 34  member corporation
  tf=   64 df= 25  nyari confederation
  tf=   63 df= 33  cgc security
  tf=   60 df= 19  wampa robotics
  tf=   58 df= 42  across haven
  tf=   57 df= 46  classification open
  tf=   57 df= 15  cgc infrastructure
```


## Top content trigrams (df 3..80)
```
  tf=   97 df= 65  haven island special
  tf=   97 df= 65  island special economic
  tf=   97 df= 65  special economic zone
  tf=   52 df= 19  genesis labs somatic
  tf=   48 df= 23  billion phi credits
  tf=   47 df= 26  million phi credits
  tf=   46 df= 31  chief executive officer
  tf=   42 df=  8  fiscal year pce
  tf=   38 df= 11  genesis labs clinical
  tf=   35 df= 12  biological growth enhancers
  tf=   33 df= 22  group cyanite industries
  tf=   32 df= 19  cgc member corporations
  tf=   32 df= 21  phyrexis group cyanite
  tf=   31 df= 11  cgc security directorate
  tf=   30 df=  9  biological growth enhancer
  tf=   28 df= 21  cyanite industries one
  tf=   28 df= 21  industries one network
  tf=   26 df=  7  cgc infrastructure committee
  tf=   26 df= 12  second-generation biological growth
  tf=   23 df= 10  maritime defense command
  tf=   22 df= 11  edge research project
  tf=   22 df= 21  distribution cgc member
  tf=   22 df= 19  cgc member representatives
  tf=   22 df= 11  west haven port
  tf=   20 df= 11  zonnon maritime defense
  tf=   20 df= 10  labs somatic clinic
  tf=   20 df=  4  maritime security division
  tf=   19 df= 19  internal memorandum classification
  tf=   19 df= 14  cgc member corporation
  tf=   18 df=  9  nguyen anh mai
```
