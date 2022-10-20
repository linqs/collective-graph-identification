#!/bin/python3
# Converts .tab files to data files for psl.
#
# Example:
#   ./tab_to_predicate.py <*.tab> (predicate_file.txt)
#   or
#   python3 tab_to_predicate.py <*.tab> (predicate_file.txt)

import sys # for arguments
import pandas as pd
import re

# used for extracting feature name
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
            # FIXME: make a case for NODE, DIRECTED, and UNDIRECTED
            # print('i is: ', i)
            if i == 0:
                # Skip non-useful first line
                print("Header: ", row)
            elif i == 1:
                # Prepare dataframe column labels
                tokens = row.split()
                if len(tokens) == 1:
                    print("This is not a NODE file, so don't load this row")
                else:
                    features = ["id"] + [get_feature_tuple(feature)[1] for feature in tokens]
                    node_data = pd.DataFrame(columns=features)
            else:
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


# Defining main function
def main():
    print("hey there")
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    attribute_table = load_table(sys.argv[1])
    print(attribute_table)
  
  
# Using the special variable 
# __name__
if __name__=="__main__":
    main()
