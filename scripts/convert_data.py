#!/bin/python3
# Converts .tab files (from namata-kdd-11) into data files for PSL.

import re
import itertools # for cross products when filling in a full PSL dataset

import pandas as pd
import numpy as np

# Full Graph
FILE_GROUND_TRUTH_EMAIL_NODES         = '../c3/namata-kdd11-data/enron/enron-samples-lowunk/outputgraph/enron.NODE.email.tab'
FILE_GROUND_TRUTH_COREF_EDGES         = '../c3/namata-kdd11-data/enron/enron-samples-lowunk/outputgraph/enron.UNDIRECTED.coref.tab'
FILE_GROUND_TRUTH_MANAGES_EDGES       = '../c3/namata-kdd11-data/enron/enron-samples-lowunk/outputgraph/enron.UNDIRECTED.email-submgr.tab'
FILE_GROUND_TRUTH_COMMUNICATION_EDGES = '../c3/namata-kdd11-data/enron/enron-samples-lowunk/outputgraph/enron.DIRECTED.sentto.tab'

# Targets
FILE_SAMPLE_EMAIL_NODES   ='../c3/namata-kdd11-data/enron/enron-samples-lowunk/enron-sample-lowunk-6of6/sample-enron.NODE.email.tab'
FILE_SAMPLE_COREF_EDGES   ='../c3/namata-kdd11-data/enron/enron-samples-lowunk/enron-sample-lowunk-6of6/sample-enron.UNDIRECTED.coref.tab'
FILE_SAMPLE_MANAGES_EDGES ='../c3/namata-kdd11-data/enron/enron-samples-lowunk/enron-sample-lowunk-6of6/sample-enron.UNDIRECTED.email-submgr.tab'

# assigns types to each column
def resolve_column_type(table):
    for column in table.columns:
        if column in {'id', 'email', 'alt_email', 'other_email' , 'numsent', 'numreceived', 'numexchanged'}:
            table[column] = table[column].astype(str).astype(float).astype(int)
        # convert bag-of-words columns to floats (since ints won't take NaNs)
        elif re.match("w-", column):
            table[column] = table[column].astype(str).astype(float)

# extracts feature name from an element in a raw tab row
# returns: tuple (feature_name, feature_value, optional_value)
def get_feature_tuple(feature):
    feature_data = re.split(r"[:=]", feature)
    return feature_data
    

# loads the *.tab files into a Pandas Dataframe.
# returns: pd.DataFrame(columns=features)
def load_table(filename):

    # initialize the pandas dataframe
    node_data = pd.DataFrame()


    with open(filename) as infile:
        i = 0
        row_list = []
        for row in infile:
    

            # Skip the 0th (non-useful) line.
            if i == 1:
                # Prepare dataframe column labels
                tokens = row.split()
                if len(tokens) == 1:
                    print("This is not a NODE file, so don't load this row")
                else:  
                    features = ["id"] + [get_feature_tuple(feature)[1] for feature in tokens]
                    node_data = pd.DataFrame(columns=features)
            elif i > 1:
          
                # this is to help the function generalize among the NODE and EDGE files.
                # EDGE files have a "|" character, which needs to be removed for proper feature decoupling
                row = re.sub(r'\|','', row)
            
                tokens = row.split()

                # the first token doesn't need splitting
                row_dict = {'id':tokens[0]}
                row_dict.update({get_feature_tuple(token)[0]:get_feature_tuple(token)[1] for token in tokens[1:]})
                row_list.append(row_dict)
        
            i += 1
        
        # Fill in rows
        node_data = pd.concat([node_data, pd.DataFrame(row_list)], ignore_index=True)

    return node_data

# Takes a table and fills the missing pairs and values to specify a full, sufficient set
# So far it only works with binary predicates
def fill_observed_missing_possibilities(table, arguments, values):
    total_possibilities = set(itertools.product(list(table[arguments[0]]), values))
    already_observed_possibilities = set((table.loc[index][arguments[0]], table.loc[index][arguments[1]]) for index in table.index)

    missing_possibilities = total_possibilities - already_observed_possibilities
    row_list = []
    for arg_0, arg_1 in missing_possibilities:
        row_dict = {arguments[0]:arg_0, arguments[1]:arg_1, arguments[2]:0 }
        row_list.append(row_dict)
        
    return pd.concat([table, pd.DataFrame(row_list)], ignore_index=True)

def process_email_nodes():
    email_nodes = load_table(FILE_GROUND_TRUTH_EMAIL_NODES)
    # remove the (unnecessary) second to last column (it came from an ambiguous parse splits)
    email_nodes.drop('other,manager,specialist,director,executive', axis=1, inplace=True)
    resolve_column_type(email_nodes)
    
    # print(email_nodes.dtypes)
    # Grab necessary columns, in preparation for dumping the whole ground truth data
    email_nodes_data = email_nodes[['id','title']].copy()

    # convert titles to integers, so PSL can ground faster
    title_map = {"other": 0, "manager": 1, "specialist": 2, "director": 3, "executive": 4}

    email_nodes_data = email_nodes_data.replace({'title': title_map})
    email_nodes_data['exists'] = 1.0

    full_set_email_has_label_data = fill_observed_missing_possibilities(email_nodes_data, ['id', 'title', 'exists'], list(title_map.values()))
    # print(full_set_email_has_label_data)
    full_set_email_has_label_data.to_csv('EmailHasLabel_data.txt', sep ='\t', index=False, header=False, columns=['id', 'title', 'exists'])

def main():
    process_email_nodes()

if __name__=="__main__":
    main()
