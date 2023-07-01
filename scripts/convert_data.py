#!/bin/python3
# Converts .tab files (from namata-kdd-11) into data files for PSL.

import os
import re
import itertools # for cross products when filling in a full PSL dataset

import pandas as pd
import numpy as np

# Full Graph
FILE_GROUND_TRUTH_EMAIL_NODES         = '../c3/namata-kdd11-data/enron/enron-samples-lowunk/outputgraph/enron.NODE.email.tab'
FILE_GROUND_TRUTH_COREF_EDGES         = '../c3/namata-kdd11-data/enron/enron-samples-lowunk/outputgraph/enron.UNDIRECTED.coref.tab'
FILE_GROUND_TRUTH_MANAGES_EDGES       = '../c3/namata-kdd11-data/enron/enron-samples-lowunk/outputgraph/enron.UNDIRECTED.email-submgr.tab'
FILE_GROUND_TRUTH_COMMUNICATION_EDGES = '../c3/namata-kdd11-data/enron/enron-samples-lowunk/outputgraph/enron.DIRECTED.sentto.tab'

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

# Outputs the whole ground truth and splits from the C3 data.
def process_email_nodes():
    # Get ground truth.
    email_nodes = load_table(FILE_GROUND_TRUTH_EMAIL_NODES)
    # Remove the (unnecessary) second to last column (it came from an ambiguous parse splits).
    email_nodes.drop('other,manager,specialist,director,executive', axis=1, inplace=True)
    resolve_column_type(email_nodes)
    
    # Grab necessary columns, in preparation for dumping the whole ground truth data.
    email_nodes_data = email_nodes[['id','title']].copy()

    # Convert titles to integers, so PSL can ground faster.
    title_map = {"other": 0, "manager": 1, "specialist": 2, "director": 3, "executive": 4}

    email_nodes_data = email_nodes_data.replace({'title': title_map})
    email_nodes_data['exists'] = 1.0

    full_set_email_has_label_data = fill_observed_missing_possibilities(email_nodes_data, ['id', 'title', 'exists'], list(title_map.values()))
    print("outputting full set for node labeling: ./EmailHasLabel_data.txt")
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
    exists_map = {"NOTEXIST": 0.0, "EXIST": 1.0}
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

    print("outputting full set for entity resolution: ./CoRef_data.txt")
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

def main():
    process_email_nodes()
    process_CoRef_edges()

if __name__=="__main__":
    main()
