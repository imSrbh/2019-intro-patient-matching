import recordlinkage  as rl
import numpy as np
import pandas as pd
import warnings
import recordlinkage
from recordlinkage.index import Block
from recordlinkage.datasets import load_febrl4,load_febrl3,load_febrl2,load_febrl1
# from recordlinkage.preprocessing import phonetic clean
from recordlinkage.preprocessing import clean, phonetic

warnings.filterwarnings('ignore')

import recordlinkage
ecm = recordlinkage.ECMClassifier()

# file to deduplicate
IMPORT_FILE_TO_DEDUPLICATE = '/home/saurabh/Projects/Healthcare/AI_patient-matching/data/dataset_febrl3.csv'
df_a = pd.read_csv(IMPORT_FILE_TO_DEDUPLICATE)
df_a = df_a.set_index('rec_id')

print("Total number of records:", len(df_a))
df_a.sort_values('surname').head()


def _preprocessing(df_a):
    df_a = df_a.copy()
    df_a['given_name'] = clean(df_a['given_name'])
    df_a['surname'] = clean(df_a['surname'])
    df_a['date_of_birth'] = pd.to_datetime(df_a['date_of_birth'],format='%Y%m%d', errors='coerce')
    df_a['YearB'] = df_a['date_of_birth'].dt.year.astype('Int64') 
    df_a['MonthB'] = df_a['date_of_birth'].dt.month.astype('Int64') 
    df_a['DayB'] = df_a['date_of_birth'].dt.day.astype('Int64')  
    df_a['metaphone_given_name'] = phonetic(df_a['given_name'], method='metaphone')
    df_a['metaphone_surname'] = phonetic(df_a['surname'], method='metaphone')  
    
    return df_a

df_a= _preprocessing(df_a)
df_a.info


###### Blocking
def _blocking(df_a):
    df_a = df_a.copy()
    indexer = rl.Index()
    # soundex firstname, methapone surname, exact date of birth
    indexer.add(Block(['metaphone_given_name','metaphone_surname','date_of_birth']))
    # soundex firstname , day of birth
    indexer.add(Block(['metaphone_given_name','DayB']))
    #soundex firstname , month of birth
    indexer.add(Block(['metaphone_given_name','MonthB']))
    # metaphone surname, year of birth 
    indexer.add(Block(['metaphone_surname','YearB']))
    # ssn
    indexer.add(Block(['soc_sec_id']))

    candidate_record_pairs = indexer.index(df_a)

    return candidate_record_pairs

candidate_record_pairs =_blocking(df_a)
print(candidate_record_pairs)
print("Number of candidate record pairs :", len(candidate_record_pairs))




########## Comparison
def _comparaison(candidate_record_pairs):  
    df_candidates = candidate_record_pairs.copy() 
    compare_cl = rl.Compare() 
    compare_cl.string('given_name', 'given_name', method='jarowinkler', label='given_name')  
    compare_cl.string('surname', 'surname', method='jarowinkler',label='surname')  
    compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')  
    compare_cl.exact('soc_sec_id', 'soc_sec_id', label='soc_sec_id')  
    compare_cl.string('address_1', 'address_1', method ='levenshtein' , label='address_1')
    compare_cl.string('address_2', 'address_2', method ='levenshtein' , label='address_2')  
    compare_cl.string('suburb', 'suburb', method ='levenshtein', label='suburb')
    compare_cl.exact('postcode', 'postcode', label='postcode')
    compare_cl.exact('state', 'state', label='state')
    features = compare_cl.compute(df_candidates, df_a)
    
    return features


features = _comparaison(candidate_record_pairs)
# print(features)

# ecm = recordlinkage.ECMClassifier()
# matches = ecm.fit_predict(features)

ecm = recordlinkage.ECMClassifier()
matches = ecm.fit(features)
p = ecm.prob(features)
# p = ecm.predict
p.tail(50)

"""

##########Get Output######################

input_rec = 'rec-10-dup-1'
matches_df = matches.to_frame()
print(matches_df.head(50))
matches_df = matches_df[matches_df.rec_id_1 ==input_rec]
print("shape:::",matches_df.shape)

rec_id_1_value = 'rec-10-dup-1'
column_name = 'rec_id_2'

# create a subset of the DataFrame where rec_id_1 == rec_id_1_value
subset = matches_df.loc[rec_id_1_value]

# access the values of the desired column in the subset using the .values attribute
values = subset[column_name].values

print("!!!!!!!Matches!!!!!!!!")
# iterate over the values and print them
for value in values:
    # print("!!!!!!!!!!!!!!!!!!")
    
    print(value)

"""
