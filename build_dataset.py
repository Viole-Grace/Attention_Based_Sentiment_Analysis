import numpy as np
import pandas as pd
import sys

def convert_to_csv(filename, data):

    return data.to_csv("{}.csv".format(filename), index=False)

def read_text_files(filename):

    data = pd.read_csv('sentiment labelled sentences/{}.txt'.format(filename), sep='\t', header=None)
    # print(data.head(5))
    # print(data.shape)
    return data

def combine_files(file1, file2):

    new_data = file1.append(file2, ignore_index =True)
    new_data.rename(columns={0:'Review',1:'Sentiment'}, inplace=True)
    # print(new_data.columns)
    # print(new_data.head(5))
    return new_data

amazon_rev = read_text_files('amazon_cells_labelled')
yelp_rev = read_text_files('yelp_labelled')

combined_rev = combine_files(amazon_rev, yelp_rev)
convert_to_csv("Combined Reviews",combined_rev)