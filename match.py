from IPython.display import display
import warnings
import numpy as np
import pandas as pd
import recordlinkage 
from recordlinkage.index import Block
from recordlinkage.preprocessing import phonetic
warnings.filterwarnings('ignore')
#read the data

input = pd.read_csv('/home/saurabh/Projects/Healthcare/2_csvs/data/input.csv')
input1 = pd.read_csv('/home/saurabh/Projects/Healthcare/2_csvs/data/input.csv')
source = pd.read_csv('/home/saurabh/Projects/Healthcare/2_csvs/data/master_table.csv')
source = source.set_index('rec_id')


# convert date of birth as string in input table
input['date_of_birth'] = pd.to_datetime(input['date_of_birth'],format='%Y%m%d', errors='coerce')
input['YearB'] = input['date_of_birth'].dt.year.astype('Int64') 
input['MonthB'] = input['date_of_birth'].dt.month.astype('Int64') 
input['DayB'] = input['date_of_birth'].dt.day.astype('Int64') 

input['metaphone_given_name'] = phonetic(input['given_name'], method='metaphone')
input['metaphone_surname'] = phonetic(input['surname'], method='metaphone')

# convert date of birth as string in source table
source['date_of_birth'] = pd.to_datetime(source['date_of_birth'],format='%Y%m%d', errors='coerce')
source['YearB'] = source['date_of_birth'].dt.year.astype('Int64') 
source['MonthB'] = source['date_of_birth'].dt.month.astype('Int64') 
source['DayB'] = source['date_of_birth'].dt.day.astype('Int64') 

source['metaphone_given_name'] = phonetic(source['given_name'], method='metaphone')
source['metaphone_surname'] = phonetic(source['surname'], method='metaphone')




indexer = recordlinkage.Index()

indexer.block(left_on=['metaphone_given_name','metaphone_surname','date_of_birth'], 
              right_on=['metaphone_given_name','metaphone_surname','date_of_birth'])
candidate_record_pairs = indexer.index( input, source)

print("Number of record pairs :",len(candidate_record_pairs))
candidate_record_pairs.to_frame(index=False)



compare_cl = recordlinkage.Compare()
compare_cl.string('given_name', 'given_name', method='jarowinkler', threshold = 0.85, label='given_name')
compare_cl.string('surname', 'surname', method='jarowinkler',threshold = 0.85, label='surname')
compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
compare_cl.exact('soc_sec_id', 'soc_sec_id', label='soc_sec_id')
compare_cl.string('address_1', 'address_1', method ='levenshtein' ,threshold = 0.85, label='address_1')
compare_cl.string('address_2', 'address_2', method ='levenshtein' ,threshold = 0.85, label='address_2')
compare_cl.string('suburb', 'suburb', method ='levenshtein' ,threshold = 0.85, label='suburb')
compare_cl.exact('postcode', 'postcode', label='postcode')
compare_cl.exact('state', 'state', label='state')

features = compare_cl.compute(candidate_record_pairs, input, source)
# print(features.head(50))


ecm = recordlinkage.ECMClassifier()
matches = ecm.fit(features)
p = ecm.prob(features)
# p = ecm.predict
# print(p.tail(50))
matches.to_csv('matches.csv')
import pandas as pd

# Load the CSV file with the 'rec_id' column into a DataFrame
rec_ids_df = pd.read_csv('/home/saurabh/Projects/Healthcare/2_csvs/data/master_table.csv')

# Set the 'rec_id' column as the index
# rec_ids_df.set_index('rec_id', inplace=True)

# Load the DataFrame with the MultiIndex into memory
multiindex_df = matches
                         

# Get a Series containing the 'rec_id' values corresponding to the first level of the MultiIndex
rec_id_series = rec_ids_df.set_index('index')['rec_id']

# Replace the first level of the MultiIndex with the 'rec_id' values
multiindex_df.index = pd.MultiIndex.from_tuples(
    [(rec_id_series[idx], level1) for (idx, level1) in multiindex_df.index]
)


match_table= pd.DataFrame(p)
match_table.reset_index(inplace=True)
# df2 = df1.rename(columns={'level_0': 'input_index'}, {'rec_id': 'match_index'}, {0: 'prob'})

match_table = match_table.rename(columns={'level_0': 'input_index', 0: 'prob'})
match_table['Status'] = 'Unsure'

# set conditions for 'Match' column based on 'Value' column
match_table.loc[match_table['prob'] > 0.8, 'Status'] = 'Duplicate'
# df1.to_csv('master_table.csv')
match_table.loc[match_table['prob'] < 0.3, 'Status'] = 'Unsure'



print(match_table.head())
print(match_table.head())

# print(match_table.columns)

input_ = input1.reset_index()
source_ = source.reset_index()

# print(source_.columns)
# print(input_.columns)


merged_df = pd.merge(match_table, input_, left_on='input_index', right_on='index', how='inner')

new_indexes = input_[~input_['index'].isin(merged_df['index'].unique())]['index']

# Get the rows in input_ with the new indexes
new_rows = input_[input_['index'].isin(new_indexes)]

# Append the new rows to merged_df
merged_df = merged_df.append(new_rows, ignore_index=True)
merged_df['Status'].fillna('Unique', inplace=True)
merged_df['prob'].fillna('-', inplace=True)
merged_df.drop(columns=['input_index'], inplace=True)
merged_df['date_of_birth'] = merged_df.apply(lambda row: str(row['YearB']) + '-' + str(row['MonthB']) + '-' + str(row['DayB']), axis=1)

print(merged_df)

# match_df = pd.DataFrame()
# Fetch rows from source_ table for the matching rec_id values
match_df = source_.loc[source_['rec_id'].isin(match_table['rec_id'])].reset_index(drop=True)

# Print the resulting dataframe
# print(match_df)
# Update values in the merged DataFrame
# merged_df['new_column'] = 'new_value'

# # def get_output_table(input_, source_, match_table):
#     # Get the indices of the matched records
# for i in match_table['input_index']:
#     print(i)
#     for j in input_['index']:
#         print("j>>>>>>:",j)
#         if(i == j):
#             input_['Prob']= match_table['prob'][0]
#             input_['Status'] = match_table['Status']
#         else:
#             print("NoMatch")
#         print(input_.head())

# output_df = pd.DataFrame()
# for i in match_table['rec_id']:
#     print(i)
#     row = source.loc[i]
#     # Append row to new_df
#     output_df = output_df.append(row, ignore_index=True)
    # for j in source_['rec_id']:
# print(output_df.head())


    # return output_table    

# output_table = get_output_table(input_, source_, match_table)

# print(output_table)


# in_diff_match_uniq = input_[~input_['index'].isin(match_table['index'])]
# print(in_diff_match_uniq)


    # Print the resulting dataframe
    # print(match_df)


    # # def get_output_table(input_, source_, match_table):
    #     # Get the indices of the matched records
    # for i in match_table['input_index']:
    #     # print(i)
    #     for j in input_['index']:
    #         if (i == j):
    #             input_['Prob']= match_table['prob']
    #             input_['Status'] = match_table['Status']
    #     # print(input_.reset_index())

    # output_df = pd.DataFrame()
    # for i in match_table['rec_id']:
    #     print(i)
    #     row = source.loc[i]
    #     # Append row to new_df
    #     output_df = output_df.append(row, ignore_index=True)
    #     # for j in source_['rec_id']:
    # print(output_df.head())

    # print(match_table)