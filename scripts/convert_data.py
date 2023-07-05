#!/bin/python3
# Converts .tab files (from namata-kdd-11) into data files for PSL.

import os
import re
import itertools # for cross products when filling in a full PSL dataset

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from strsimpy.qgram import QGram
from scipy.spatial import distance
from matplotlib import pyplot as plt

# Full Graph
FILE_GROUND_TRUTH_EMAIL_NODES         = '../c3/namata-kdd11-data/enron/enron-samples-lowunk/outputgraph/enron.NODE.email.tab'
FILE_GROUND_TRUTH_COREF_EDGES         = '../c3/namata-kdd11-data/enron/enron-samples-lowunk/outputgraph/enron.UNDIRECTED.coref.tab'
FILE_GROUND_TRUTH_MANAGES_EDGES       = '../c3/namata-kdd11-data/enron/enron-samples-lowunk/outputgraph/enron.UNDIRECTED.email-submgr.tab'
FILE_GROUND_TRUTH_COMMUNICATION_EDGES = '../c3/namata-kdd11-data/enron/enron-samples-lowunk/outputgraph/enron.DIRECTED.sentto.tab'

# Convert titles to integers, so PSL can ground faster.
title_map = {"other": 0, "manager": 1, "specialist": 2, "director": 3, "executive": 4}
# convert existence column to boolean, so PSL can ground faster
exists_map = {"NOTEXIST": 0.0, "EXIST": 1.0}

# Assigns types to each column.
def resolve_column_type(table):
    for column in table.columns:
        if column in {'id', 'email', 'other_email', 'other_email' , 'numsent', 'numreceived', 'numexchanged'}:
            table[column] = table[column].astype(str).astype(float).astype(int)
        # convert bag-of-words columns to floats (since ints won't take NaNs)
        elif re.match("w-", column):
            table[column] = table[column].astype(str).astype(float)

# Extracts feature name from an element in a raw tab row.
# Returns: tuple (feature_name, feature_value, optional_value).
def get_feature_tuple(feature):
    feature_data = re.split(r"[:=]", feature)
    return feature_data

def replace_rightmost(pattern, replacement, text):
    # Find all occurrences of the pattern in the text
    matches = re.findall(pattern, text)

    if len(matches) > 0:
        # Find the index of the rightmost occurrence
        rightmost_index = text.rindex(matches[-1])

        # Replace the rightmost occurrence
        text = text[:rightmost_index] + re.sub(pattern, replacement, text[rightmost_index:], count=1)

    return text
    
# Loads the *.tab files into a Pandas Dataframe.
# Returns: pd.DataFrame(columns=features)
def load_table(filename):

    # Initialize the Pandas DataFrame.
    node_data = pd.DataFrame()


    with open(filename) as infile:
        i = 0
        row_list = []
        for row in infile:
    

            # Skip the 0th (non-useful) line.
            if i == 1:
                # Prepare dataframe column labels.
                tokens = row.split()
                if len(tokens) == 1:
                    # print("This is not a NODE file, so don't load this row")
                    i += 1
                    continue
                else:  
                    features = ["id"] + [get_feature_tuple(feature)[1] for feature in tokens]
                    node_data = pd.DataFrame(columns=features)
            elif i > 1:
          
                # This is to help the function generalize among the NODE and EDGE files.
                # EDGE files have a "|" character, which needs to be removed for proper feature decoupling.
                row = re.sub(r'\|','', row)

				# For the UNDIRECTED edge files, we need to rename one of the columns due to key collision
                row = replace_rightmost("email:", "other_email:", row)
            
                tokens = row.split()

                # the first token doesn't need splitting
                row_dict = {'id':tokens[0]}
                row_dict.update({get_feature_tuple(token)[0]:get_feature_tuple(token)[1] for token in tokens[1:]})
                row_list.append(row_dict)
        
            i += 1
        
        # Fill in rows.
        node_data = pd.concat([node_data, pd.DataFrame(row_list)], ignore_index=True)

    return node_data

# Takes a table and fills the missing pairs and values to specify a full, sufficient set.
# So far it only works with binary predicates.
def fill_observed_missing_possibilities(table, arguments, values):
    total_possibilities = set(itertools.product(list(table[arguments[0]]), values))
    already_observed_possibilities = set((table.loc[index][arguments[0]], table.loc[index][arguments[1]]) for index in table.index)

    missing_possibilities = total_possibilities - already_observed_possibilities
    row_list = []
    for arg_0, arg_1 in missing_possibilities:
        row_dict = {arguments[0]:arg_0, arguments[1]:arg_1, arguments[2]:0 }
        row_list.append(row_dict)
        
    return pd.concat([table, pd.DataFrame(row_list)], ignore_index=True)

def output_to_dir(df, outdir, outname):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    fullname = os.path.join(outdir, outname)
    
    print("outputing split to: ", fullname) 
    df.to_csv(fullname, sep = '\t', index = False, header = False)

def train_local_node_labels(email_nodes_obs, email_nodes_truth):
    train_x = email_nodes_obs.drop(['id', 'emailaddress', 'title', 'numsent', 'numreceived', 'numexchanged'], axis = 1).fillna(0)
    train_y = email_nodes_obs['title']

    test_x = email_nodes_truth.drop(['id', 'emailaddress', 'title', 'numsent', 'numreceived', 'numexchanged'], axis = 1).fillna(0)
    test_y = email_nodes_truth['title']
    classifier = LogisticRegression(max_iter=300)
    classifier.fit(train_x, train_y)
    # predictions = classifier.predict(test_x)

    # Use probabilities for PSL observed data.
    local_EmailHasTitle_probabilities = classifier.predict_proba(test_x)
    local_EmailHasTitle_obs = pd.DataFrame()
    row_list = []
    # build a table
    for index, probabilities in enumerate(local_EmailHasTitle_probabilities):
        for class_index, probability in enumerate(probabilities):
            row_dict = {'id': email_nodes_truth.iloc[index]['id'], 'title': title_map[classifier.classes_[class_index]], 'exists': probability}
            row_list.append(row_dict)
    
    local_EmailHasTitle_obs = pd.concat([local_EmailHasTitle_obs, pd.DataFrame(row_list)], ignore_index=True)
    return local_EmailHasTitle_obs

# Outputs the whole ground truth and splits from the C3 data.
def process_email_nodes():
    # Get ground truth.
    email_nodes = load_table(FILE_GROUND_TRUTH_EMAIL_NODES)
    # Remove the (unnecessary) second to last column (it came from an ambiguous parse splits).
    email_nodes.drop('other,manager,specialist,director,executive', axis=1, inplace=True)
    resolve_column_type(email_nodes)
    
    # Grab necessary columns, in preparation for dumping the whole ground truth data.
    email_nodes_data = email_nodes[['id','title']].copy()


    email_nodes_data = email_nodes_data.replace({'title': title_map})
    email_nodes_data['exists'] = 1.0

    full_set_email_has_label_data = fill_observed_missing_possibilities(email_nodes_data, ['id', 'title', 'exists'], list(title_map.values()))
    print("outputting full (unsplit) set for node labeling: ./EmailHasLabel_data.txt")
    full_set_email_has_label_data.to_csv('EmailHasLabel_data.txt', sep ='\t', index=False, header=False, columns=['id', 'title', 'exists'])

    # Get targets, calculate splits for PSL predicates.
    for i in range(1, 7):
        # Targets
        SPLIT_NUM = i
        FILE_SAMPLE_EMAIL_NODES   = f'../c3/namata-kdd11-data/enron/enron-samples-lowunk/enron-sample-lowunk-{SPLIT_NUM}of6/sample-enron.NODE.email.tab'
        # Grab the sample from the original experiment, this will allow us to calculate observations and targets.
        sample_email_nodes = load_table(FILE_SAMPLE_EMAIL_NODES)

        # Remove the (unnecessary) second to last column (it came from an ambiguous parse splits).
        sample_email_nodes.drop('other,manager,specialist,director,executive', axis=1, inplace=True)
        resolve_column_type(sample_email_nodes)

        # Split data into observed and targets (AKA train and test).
        email_nodes_obs = email_nodes[email_nodes['id'].isin(sample_email_nodes[sample_email_nodes['title'].notna()]['id'])]
        email_nodes_truth = email_nodes[email_nodes['id'].isin(sample_email_nodes[sample_email_nodes['title'].isna()]['id'])]

        # Grab the necessary columns.
        email_has_label_obs = email_nodes_obs[['id','title']].copy()
        email_has_label_truth = email_nodes_truth[['id','title']].copy()

        # Convert titles to integers, so PSL can ground faster.
        email_has_label_obs = email_has_label_obs.replace({'title': title_map})
        email_has_label_truth = email_has_label_truth.replace({'title': title_map})

        # Add in an existence column.
        email_has_label_obs['exists'] = 1.0
        email_has_label_truth['exists'] = 1.0

        # Add in the the non existent observations.
        full_set_email_has_label_obs = fill_observed_missing_possibilities(email_has_label_obs, ['id', 'title', 'exists'], list(title_map.values()))
        full_set_email_has_label_truth = fill_observed_missing_possibilities(email_has_label_truth, ['id', 'title', 'exists'], list(title_map.values()))

        # Outputs splits to file.
        # PSL splits start at 00, not 1.
        outdir = './enron/' + str(i-1).zfill(2) + '/eval'
        output_to_dir(full_set_email_has_label_obs, outdir, 'EmailHasLabel_obs.txt')
        output_to_dir(full_set_email_has_label_truth, outdir, 'EmailHasLabel_truth.txt')

        local_EmailHasTitle_obs = train_local_node_labels(email_nodes_obs, email_nodes_truth)
        output_to_dir(local_EmailHasTitle_obs, outdir, 'Local_EmailHasLabel_obs.txt')

def train_local_CoRef_edges(full_set_coref_edges_obs, coref_truth, email_nodes):
    node_to_email = dict(zip(email_nodes['id'], email_nodes['emailaddress']))

    qgram = QGram(1)
    train_x = full_set_coref_edges_obs.copy()
    train_y = full_set_coref_edges_obs['exists'].copy()
    
    train_x['address_similarity'] = 0.0
    train_x['bow_cosine_similarity'] = 0.0
    train_x['bow_jaccard_similarity'] = 0.0
    
    emailID_to_rowID = {key: value for key, value in zip(list(email_nodes['id']), list(email_nodes.index))}
    for index, row in train_x.iterrows():
        string_similarity = qgram.distance(email_nodes.iloc[emailID_to_rowID[row['email']]]['emailaddress'], email_nodes.iloc[emailID_to_rowID[row['other_email']]]['emailaddress'])
        train_x.loc[index, 'address_similarity'] = string_similarity
    
        bow_cosine_similarity = distance.cosine(np.nan_to_num(list(email_nodes.iloc[emailID_to_rowID[row['email']]][5:-1])), np.nan_to_num(list(email_nodes.iloc[emailID_to_rowID[row['other_email']]][5:-1])))
        train_x.loc[index, 'bow_cosine_similarity'] = bow_cosine_similarity
    
        bow_jaccard_similarity = distance.jaccard(np.nan_to_num(list(email_nodes.iloc[emailID_to_rowID[row['email']]][5:-1])), np.nan_to_num(list(email_nodes.iloc[emailID_to_rowID[row['other_email']]][5:-1])))
        train_x.loc[index, 'bow_jaccard_similarity'] = bow_jaccard_similarity  

    train_x = train_x.drop(['email', 'other_email', 'exists'], axis = 1)

    test_x = coref_truth.copy()
    test_y = coref_truth['exists'].copy()
    
    test_x['address_similarity'] = 0.0
    test_x['bow_cosine_similarity'] = 0.0
    test_x['bow_jaccard_similarity'] = 0.0
    
    for index, row in test_x.iterrows():
        string_similarity = qgram.distance(email_nodes.iloc[emailID_to_rowID[row['email']]]['emailaddress'], email_nodes.iloc[emailID_to_rowID[row['other_email']]]['emailaddress'])
        test_x.loc[index, 'address_similarity'] = string_similarity
    
        bow_cosine_similarity = distance.cosine(np.nan_to_num(list(email_nodes.iloc[emailID_to_rowID[row['email']]][5:-1])), np.nan_to_num(list(email_nodes.iloc[emailID_to_rowID[row['other_email']]][5:-1])))
        test_x.loc[index, 'bow_cosine_similarity'] = bow_cosine_similarity
    
        bow_jaccard_similarity = distance.jaccard(np.nan_to_num(list(email_nodes.iloc[emailID_to_rowID[row['email']]][5:-1])), np.nan_to_num(list(email_nodes.iloc[emailID_to_rowID[row['other_email']]][5:-1])))
        test_x.loc[index, 'bow_jaccard_similarity'] = bow_jaccard_similarity  
            
    test_x = test_x.drop(['email', 'other_email', 'exists'], axis = 1)

    classifier = LogisticRegression()
    classifier.fit(train_x, train_y)

    local_CoRef_probabilities = classifier.predict_proba(test_x)
    local_CoRef_obs = pd.DataFrame()
    row_list = []
    # build a table
    for index, probabilities in enumerate(local_CoRef_probabilities):
        row_dict = {'email': int(coref_truth.iloc[index]['email']), 'other_email': int(coref_truth.iloc[index]['other_email']), 'exists': probabilities[1]}
        row_list.append(row_dict)
        #print(email_nodes_truth.iloc[index]['id'], "\t", title_map[classifier.classes_[class_index]], "\t", probability)
    
    local_CoRef_obs = pd.concat([local_CoRef_obs, pd.DataFrame(row_list)], ignore_index=True)
    return local_CoRef_obs


# Outputs the whole ground truth and splits from the C3 data.
def process_CoRef_edges():
    # Need to process email nodes so we can later calculate the 'blocked' edges from the c3 datasets.
    email_nodes = load_table(FILE_GROUND_TRUTH_EMAIL_NODES)
    # Remove the (unnecessary) second to last column (it came from an ambiguous parse splits).
    email_nodes.drop('other,manager,specialist,director,executive', axis=1, inplace=True)
    resolve_column_type(email_nodes)

    # Start loading in the CoRef edges.
    coref_edges = load_table(FILE_GROUND_TRUTH_COREF_EDGES)
    resolve_column_type(coref_edges)

    # Grab necessary columns, in preparation for dumping the whole ground truth data
    coref_edges_data = coref_edges[['email','other_email', 'exists']].copy()
    
    # convert existence column to boolean, so PSL can ground faster
    coref_edges_data = coref_edges_data.replace({'exists': exists_map})
    
    # Since it's undirected, add in the reverse edges.
    coref_edges_data_sym = coref_edges_data[['other_email', 'email', 'exists']].copy()
    coref_edges_data_sym.rename(columns = {'other_email':'email', 'email':'other_email'}, inplace = True)
    
    coref_edges_data = pd.concat([coref_edges_data, coref_edges_data_sym])
    
    # Calculated the missing edges that were blocked.
    missing_edges = {pair for pair in itertools.permutations(email_nodes['id'], 2)} - {pair for pair in zip(coref_edges_data['email'], coref_edges_data['other_email'])}
    
    # add in the missing edges
    row_list = []
    for email, other_email in missing_edges:
        row_dict = {'email':email, 'other_email':other_email, 'exists':0 }
        row_list.append(row_dict)

    full_set_coref_edges_data = pd.concat([coref_edges_data, pd.DataFrame(row_list)], ignore_index=True)

    print("outputting full (unsplit) set for entity resolution: ./CoRef_data.txt")
    full_set_coref_edges_data.to_csv('CoRef_data.txt', sep ='\t', index=False, header=False, columns=['email', 'other_email', 'exists'])

    # Get targets, calculate splits for PSL predicates.
    for i in range(1, 7):
        # Targets
        SPLIT_NUM = i
        FILE_SAMPLE_COREF_EDGES   = f'../c3/namata-kdd11-data/enron/enron-samples-lowunk/enron-sample-lowunk-{SPLIT_NUM}of6/sample-enron.UNDIRECTED.coref.tab'
        # Grab the sample from the original experiment, this will allow us to calculate observations and targets.
        sample_coref_edges = load_table(FILE_SAMPLE_COREF_EDGES)
        resolve_column_type(sample_coref_edges)

        # Split data into observed and targets (AKA train and test)
        coref_edges_obs = coref_edges[coref_edges['id'].isin(sample_coref_edges[sample_coref_edges['exists'].notna()]['id'])]
        coref_edges_truth = coref_edges[coref_edges['id'].isin(sample_coref_edges[sample_coref_edges['exists'].isna()]['id'])]

        # Grab the necessary columns
        coref_obs = coref_edges_obs[['email', 'other_email', 'exists']].copy()
        coref_truth = coref_edges_truth[['email', 'other_email', 'exists']].copy()
        
        # convert existence column to boolean, so PSL can ground faster
        coref_obs = coref_obs.replace({'exists': exists_map})
        coref_truth = coref_truth.replace({'exists': exists_map})
        
        # Since it's undirected, add in the reverse edges.
        coref_obs_sym = coref_obs[['other_email', 'email', 'exists']].copy()
        coref_truth_sym = coref_truth[['other_email', 'email', 'exists']].copy()
        
        coref_obs_sym.rename(columns = {'other_email':'email', 'email':'other_email'}, inplace = True)
        coref_truth_sym.rename(columns = {'other_email':'email', 'email':'other_email'}, inplace = True)
        
        coref_obs = pd.concat([coref_obs, coref_obs_sym], ignore_index=True)
        coref_truth = pd.concat([coref_truth, coref_truth_sym], ignore_index=True)
        
        # Calculated the missing edges that were blocked. Note the last set prevents cross contamination
        missing_edges = {pair for pair in itertools.permutations(email_nodes['id'], 2)} - {pair for pair in zip(coref_obs['email'], coref_obs['other_email'])} - {pair for pair in zip(coref_truth['email'], coref_truth['other_email'])}
        
        # add in the missing edges
        row_list = []
        for email, other_email in missing_edges:
            row_dict = {'email':email, 'other_email':other_email, 'exists':0 }
            row_list.append(row_dict)
        
        full_set_coref_edges_obs = pd.concat([coref_obs, pd.DataFrame(row_list)], ignore_index=True)

        # Outputs splits to file.
        # PSL splits start at 00, not 1.
        outdir = './enron/' + str(i-1).zfill(2) + '/eval'
        output_to_dir(full_set_coref_edges_obs, outdir, 'CoRef_obs.txt')
        output_to_dir(coref_truth, outdir, 'CoRef_truth.txt')

        local_CoRef_obs = train_local_CoRef_edges(full_set_coref_edges_obs, coref_truth, email_nodes)
        output_to_dir(local_CoRef_obs, outdir, 'Local_CoRef_obs.txt')

def train_local_manager_edges(manager_edges_obs, manager_edges_truth):
    train_x = manager_edges_obs.drop(['id', 'numexchanged', 'email', 'other_email', 'exists'], axis = 1).fillna(0)
    train_y = manager_edges_obs['exists']
    
    test_x = manager_edges_truth.drop(['id', 'numexchanged', 'email', 'other_email', 'exists'], axis = 1).fillna(0)
    test_y = manager_edges_truth['exists']

    classifier = LogisticRegression(max_iter=300)
    classifier.fit(train_x, train_y)
    local_Manages_probabilities = classifier.predict_proba(test_x)
    local_Manages_obs = pd.DataFrame()
    row_list = []
    # build a table
    for index, probabilities in enumerate(local_Manages_probabilities):
        row_dict = {'email': manager_edges_truth.iloc[index]['email'], 'other_email': manager_edges_truth.iloc[index]['other_email'], 'exists': exists_map[classifier.classes_[np.argmax(probabilities)]]}
        row_list.append(row_dict)
        #print(email_nodes_truth.iloc[index]['id'], "\t", title_map[classifier.classes_[class_index]], "\t", probability)
    
    local_Manages_obs = pd.concat([local_Manages_obs, pd.DataFrame(row_list)])
    # Since it's undirected, add in the reverse edges.
    local_Manages_obs_sym = local_Manages_obs[['other_email', 'email', 'exists']].copy()
    
    local_Manages_obs_sym.rename(columns = {'other_email':'email', 'email':'other_email'}, inplace = True)
    
    local_Manages_obs = pd.concat([local_Manages_obs, local_Manages_obs_sym])
    return local_Manages_obs


# Outputs the whole ground truth and splits from the C3 data.
def process_manager_edges():
    # Need to process email nodes so we can later calculate the 'blocked' edges from the c3 datasets.
    email_nodes = load_table(FILE_GROUND_TRUTH_EMAIL_NODES)
    # Remove the (unnecessary) second to last column (it came from an ambiguous parse splits).
    email_nodes.drop('other,manager,specialist,director,executive', axis=1, inplace=True)
    resolve_column_type(email_nodes)

    manager_edges = load_table(FILE_GROUND_TRUTH_MANAGES_EDGES)
    # FIXME: can probably omit this line.
    manager_edges.drop('NOTEXIST,EXIST', axis=1, inplace=True)
    resolve_column_type(manager_edges)

    # Grab necessary columns, in preparation for dumping the whole ground truth data.
    manager_edges_data = manager_edges[['email','other_email', 'exists']].copy()
    
    # Convert existence column to boolean, so PSL can ground faster.
    manager_edges_data = manager_edges_data.replace({'exists': exists_map})
    
    # Since it's undirected, add in the reverse edges.
    manager_edges_data_sym = manager_edges_data[['other_email', 'email', 'exists']].copy()
    manager_edges_data_sym.rename(columns = {'other_email':'email', 'email':'other_email'}, inplace = True)
    
    manager_edges_data = pd.concat([manager_edges_data, manager_edges_data_sym])
    
    # Calculated the missing edges that were blocked.
    missing_edges = {pair for pair in itertools.permutations(email_nodes['id'], 2)} - {pair for pair in zip(manager_edges_data['email'], manager_edges_data['other_email'])}
    
    # Add in the missing edges.
    row_list = []
    for email, other_email in missing_edges:
        row_dict = {'email':email, 'other_email':other_email, 'exists':0 }
        row_list.append(row_dict)
    
    full_set_manager_edges_data = pd.concat([manager_edges_data, pd.DataFrame(row_list)], ignore_index=True)
    print("outputting full (unsplit) set for link prediction: ./Manages_data.txt")
    full_set_manager_edges_data.to_csv('Manages_data.txt', sep ='\t', index=False, header=False, columns=['email', 'other_email', 'exists'])

    # Get targets, calculate splits for PSL predicates.
    for i in range(1, 7):
        # Targets
        SPLIT_NUM = i
        FILE_SAMPLE_MANAGES_EDGES = f'../c3/namata-kdd11-data/enron/enron-samples-lowunk/enron-sample-lowunk-{SPLIT_NUM}of6/sample-enron.UNDIRECTED.email-submgr.tab'
        # Grab the sample from the original experiment, this will allow us to calculate observations and targets.
        sample_manager_edges = load_table(FILE_SAMPLE_MANAGES_EDGES)
        resolve_column_type(sample_manager_edges)

        # Split data into observed and targets (AKA train and test)
        manager_edges_obs = manager_edges[manager_edges['id'].isin(sample_manager_edges[sample_manager_edges['exists'].notna()]['id'])]
        manager_edges_truth = manager_edges[manager_edges['id'].isin(sample_manager_edges[sample_manager_edges['exists'].isna()]['id'])]

        # Grab the necessary columns
        manages_obs = manager_edges_obs[['email', 'other_email', 'exists']].copy()
        manages_truth = manager_edges_truth[['email', 'other_email', 'exists']].copy()
        
        # convert existence column to boolean, so PSL can ground faster
        manages_obs = manages_obs.replace({'exists': exists_map})
        manages_truth = manages_truth.replace({'exists': exists_map})
        
        # Since it's undirected, add in the reverse edges.
        manages_obs_sym = manages_obs[['other_email', 'email', 'exists']].copy()
        manages_truth_sym = manages_truth[['other_email', 'email', 'exists']].copy()
        
        manages_obs_sym.rename(columns = {'other_email':'email', 'email':'other_email'}, inplace = True)
        manages_truth_sym.rename(columns = {'other_email':'email', 'email':'other_email'}, inplace = True)
        
        manages_obs = pd.concat([manages_obs, manages_obs_sym])
        manages_truth = pd.concat([manages_truth, manages_truth_sym])
        
        # Calculated the missing edges that were blocked. Note the last set prevents cross contamination
        missing_edges = {pair for pair in itertools.permutations(email_nodes['id'], 2)} - {pair for pair in zip(manages_obs['email'], manages_obs['other_email'])} - {pair for pair in zip(manages_truth['email'], manages_truth['other_email'])}
        
        # add in the missing edges
        row_list = []
        for email, other_email in missing_edges:
            row_dict = {'email':email, 'other_email':other_email, 'exists':0 }
            row_list.append(row_dict)
        
        full_set_manages_obs = pd.concat([manages_obs, pd.DataFrame(row_list)], ignore_index=True)

        # Outputs splits to file.
        # PSL splits start at 00, not 1.
        outdir = './enron/' + str(i-1).zfill(2) + '/eval'
        output_to_dir(full_set_manages_obs, outdir, 'Manages_obs.txt')
        output_to_dir(manages_truth, outdir, 'Manages_truth.txt')

        local_Manages_obs = train_local_manager_edges(manager_edges_obs, manager_edges_truth)
        output_to_dir(local_Manages_obs, outdir, 'Local_Manages_obs.txt')

def calculate_email_address_similarities():
    email_nodes = load_table(FILE_GROUND_TRUTH_EMAIL_NODES)
    # Remove the (unnecessary) second to last column (it came from an ambiguous parse splits).
    email_nodes.drop('other,manager,specialist,director,executive', axis=1, inplace=True)
    resolve_column_type(email_nodes)

    email_pairs = {pair for pair in itertools.combinations(email_nodes['id'], 2)}
    
    qgram = QGram(1)
    
    sim_email = pd.DataFrame()
    row_list = []
    
    # TODO: Optimize this with the emailID_to_rowID map.
    for pair in email_pairs:
        email_1 = email_nodes[email_nodes['id'] == pair[0]]['emailaddress'].iloc[0]
        email_2 = email_nodes[email_nodes['id'] == pair[1]]['emailaddress'].iloc[0]    
    
        string_similarity = qgram.distance(email_1, email_2)
    
        row_dict = {'email':pair[0], 'other_email':pair[1], 'qgram_sim':string_similarity}
        row_list.append(row_dict)
        
    sim_email = pd.concat([sim_email, pd.DataFrame(row_list)])

    # Since it's undirected, add in the reverse edges.
    sim_email_sym = sim_email[['other_email', 'email', 'qgram_sim']].copy()
    
    sim_email_sym.rename(columns = {'other_email':'email', 'email':'other_email'}, inplace = True)
    
    total_sim_email = pd.concat([sim_email, sim_email_sym], ignore_index=True)

    is_similar = []
    for sim in total_sim_email['qgram_sim']:
        is_similar.append(float(sim < 3))
    
    total_sim_email["is_similar"] = is_similar
    total_sim_email = total_sim_email.drop(['qgram_sim'], axis = 1)

    for i in range(0, 6):
        outdir = './enron/' + str(i).zfill(2) + '/eval'
        output_to_dir(total_sim_email, outdir, 'Sim_Email_thresh_3.txt')

def calculate_bag_of_words_similarities():
    email_nodes = load_table(FILE_GROUND_TRUTH_EMAIL_NODES)
    # Remove the (unnecessary) second to last column (it came from an ambiguous parse splits).
    email_nodes.drop('other,manager,specialist,director,executive', axis=1, inplace=True)
    resolve_column_type(email_nodes)

    email_pairs = {pair for pair in itertools.combinations(email_nodes['id'], 2)}

    sim_bow = pd.DataFrame()
    row_list = []
    
    
    emailID_to_rowID = {key: value for key, value in zip(list(email_nodes['id']), list(email_nodes.index))}
    for pair in email_pairs:
        entity_1 = email_nodes.iloc[emailID_to_rowID[pair[0]]]
        entity_2 = email_nodes.iloc[emailID_to_rowID[pair[1]]]
    
        bow_1 = entity_1[5:-1]
        bow_2 = entity_2[5:-1]

        row_dict = {'email':pair[0], 'other_email':pair[1], 'jaccard_sim_bow':distance.jaccard(list(bow_1), list(bow_2)), 'cosine_sim_bow':distance.cosine(np.nan_to_num(list(bow_1)), np.nan_to_num(list(bow_2)))}
        row_list.append(row_dict)

    sim_bow = pd.concat([sim_bow, pd.DataFrame(row_list)])
    # Since it's undirected, add in the reverse edges.
    sim_bow_sym = sim_bow[['other_email', 'email', 'jaccard_sim_bow', 'cosine_sim_bow']].copy()
    sim_bow_sym.rename(columns = {'other_email':'email', 'email':'other_email'}, inplace = True)
    
    sim_bow = pd.concat([sim_bow, sim_bow_sym])
    for i in range(0, 6):
        outdir = './enron/' + str(i).zfill(2) + '/eval/' + 'Sim_Jaccard_BOW.txt'
        print("Outputting split to: ", outdir)
        sim_bow.to_csv(outdir, sep ='\t', index=False, header=False, columns=['email', 'other_email', 'jaccard_sim_bow'])

        outdir = './enron/' + str(i).zfill(2) + '/eval/' + 'Sim_Cosine_BOW.txt'
        print("Outputting split to: ", outdir)
        sim_bow.to_csv(outdir, sep ='\t', index=False, header=False, columns=['email', 'other_email', 'cosine_sim_bow'])

# Produces communication observations and network similarities.
def calculate_network_similarities():
    email_nodes = load_table(FILE_GROUND_TRUTH_EMAIL_NODES)
    # Remove the (unnecessary) second to last column (it came from an ambiguous parse splits).
    email_nodes.drop('other,manager,specialist,director,executive', axis=1, inplace=True)
    resolve_column_type(email_nodes)

    email_pairs = {pair for pair in itertools.combinations(email_nodes['id'], 2)}

    communication_edges = load_table(FILE_GROUND_TRUTH_COMMUNICATION_EDGES)
    resolve_column_type(communication_edges)
    # Add in existence column.
    communication_edges['exists'] = 1.0
    
    # Calculated the missing edges that were blocked.
    missing_edges = {pair for pair in itertools.permutations(email_nodes['id'], 2)} - {pair for pair in zip(communication_edges['email'], communication_edges['other_email'])}
    # Add in the missing edges.
    row_list = []
    for email, other_email in missing_edges:
        row_dict = {'email':email, 'other_email':other_email, 'exists':0 }
        row_list.append(row_dict)

    full_set_communication_edges = pd.concat([communication_edges, pd.DataFrame(row_list)], ignore_index=True)

    for i in range(0, 6):
        outdir = './enron/' + str(i).zfill(2) + '/eval/' + 'Communicates.txt'
        print("Outputting split to: ", outdir)
        full_set_communication_edges.to_csv(outdir, sep ='\t', index=False, header=False, columns=['email', 'other_email', 'exists'])

    # Start loading in the CoRef edges for network similarities.
    coref_edges = load_table(FILE_GROUND_TRUTH_COREF_EDGES)
    resolve_column_type(coref_edges)

    # Grab necessary columns, in preparation for dumping the whole ground truth data
    coref_edges_data = coref_edges[['email','other_email', 'exists']].copy()
    
    # convert existence column to boolean, so PSL can ground faster
    coref_edges_data = coref_edges_data.replace({'exists': exists_map})
    
    # Since it's undirected, add in the reverse edges.
    coref_edges_data_sym = coref_edges_data[['other_email', 'email', 'exists']].copy()
    coref_edges_data_sym.rename(columns = {'other_email':'email', 'email':'other_email'}, inplace = True)
    
    coref_edges_data = pd.concat([coref_edges_data, coref_edges_data_sym])
    
    # Calculated the missing edges that were blocked.
    missing_edges = {pair for pair in itertools.permutations(email_nodes['id'], 2)} - {pair for pair in zip(coref_edges_data['email'], coref_edges_data['other_email'])}
    
    # add in the missing edges
    row_list = []
    for email, other_email in missing_edges:
        row_dict = {'email':email, 'other_email':other_email, 'exists':0 }
        row_list.append(row_dict)

    full_set_coref_edges_data = pd.concat([coref_edges_data, pd.DataFrame(row_list)], ignore_index=True)


    # prepare ground truth
    coref_map = {(int(full_set_coref_edges_data.iloc[index]['email']), int(full_set_coref_edges_data.iloc[index]['other_email'])):full_set_coref_edges_data.iloc[index]['exists'] for index in full_set_coref_edges_data.index}
    sim_network = pd.DataFrame()
    row_list = []
    
    
    # TODO: Optimize this with the emailID_to_rowID map.
    for id_1, id_2 in email_pairs:
    
        adjacent_nodes_1 = set(communication_edges[communication_edges['email'] == id_1]['other_email'])
        adjacent_nodes_2 = set(communication_edges[communication_edges['email'] == id_2]['other_email'])
    
        entity_1 = email_nodes[email_nodes['id'] == id_1]
        entity_2 = email_nodes[email_nodes['id'] == id_2]
    
        bow_1 = entity_1.iloc[0][5:-1]
        bow_2 = entity_2.iloc[0][5:-1]
    
    
        jaccard_sim =  len(adjacent_nodes_1 & adjacent_nodes_2) / len(adjacent_nodes_1 | adjacent_nodes_2 ) if len(adjacent_nodes_1 | adjacent_nodes_2) != 0 else 0
        dice_sim =  (2 * len(adjacent_nodes_1 & adjacent_nodes_2) ) / (len(adjacent_nodes_1) + len(adjacent_nodes_2)) if len(adjacent_nodes_1) + len(adjacent_nodes_2) != 0 else 0
    
    
        row_dict = {'email':id_1, 'other_email':id_2, 'jaccard_sim_network':jaccard_sim, 'dice_sim_network':dice_sim, 'jaccard_sim_bow':distance.jaccard(list(bow_1), list(bow_2)), 'cosine_sim_bow':distance.cosine(np.nan_to_num(list(bow_1)), np.nan_to_num(list(bow_2))), 'is_coref': coref_map[(id_1, id_2)]}
        row_list.append(row_dict)
    
    sim_network = pd.concat([sim_network, pd.DataFrame(row_list)])

    # Since it's undirected, add in the reverse edges.
    sim_network_sym = sim_network[['other_email', 'email', 'jaccard_sim_network', 'dice_sim_network', 'jaccard_sim_bow', 'cosine_sim_bow', 'is_coref']].copy()
    sim_network_sym.rename(columns = {'other_email':'email', 'email':'other_email'}, inplace = True)
    
    sim_network = pd.concat([sim_network, sim_network_sym], ignore_index=True)
    for i in range(0, 6):
        outdir = './enron/' + str(i).zfill(2) + '/eval/' + 'Sim_Jaccard_Network.txt'
        print("Outputting split to: ", outdir)
        sim_network.to_csv(outdir, sep ='\t', index=False, header=False, columns=['email', 'other_email', 'jaccard_sim_network'])


def process_similarity_metrics():
    calculate_email_address_similarities()
    calculate_bag_of_words_similarities()
    calculate_network_similarities()


def main():
    process_email_nodes()
    process_CoRef_edges()
    process_manager_edges()
    process_similarity_metrics()

if __name__=="__main__":
    main()
