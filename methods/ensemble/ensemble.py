import os
import shutil
import sys

import numpy as np
from scipy import sparse

import seaborn as sn
sn.set()

import pandas as pd

import bottleneck as bn

from scipy.stats import rankdata

import heapq
import math


# In[ ]:

# Config
DATA_DIR = 'ensemble_data_jester/'


import csv


def eval_ratings_custom_at_end(predicted_preference_matrix, positive_test_sample_list, negative_test_samples_list, k=10):
    hr_dist = []
    ndcg_dist = []
    for positive_test_sample, negative_test_samples, pred_user_preferences in zip(positive_test_sample_list, negative_test_samples_list, predicted_preference_matrix):
        #print(positive_test_samples)
        target_test_sample = positive_test_sample
        #print(target_test_sample)
        hr, ndcg = eval_one_rating(target_test_sample, negative_test_samples, pred_user_preferences, k)
        hr_dist.append(hr)
        ndcg_dist.append(ndcg)
        
    return np.array(hr_dist), np.array(ndcg_dist)


def eval_one_rating(a_target_item, a_negative_items, a_predictions, k=10):
    items = a_negative_items
    gtItem = a_target_item
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    #users = np.full(len(items), u, dtype = 'int32')
    predictions = a_predictions
    """
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    """
    
    for item in items:
        #print(item)
        map_item_score[item] = predictions[item]
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return hr, ndcg

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def read_baseline_data(filename):
    unique_test_movie_ids = set()
    unique_test_user_ids = set()

    raw_test_data = []

    with open(os.path.join(DATA_DIR, filename)) as csvfile:
        readCSV = csv.reader(csvfile, delimiter='\t')
        for row in readCSV:
            row_data = []
            commaPos = row[0].find(",", 0)
            userID = int(row[0][1:commaPos])
            row_data.append(userID)
            unique_test_user_ids.add(userID)
            testMovieID = int(row[0][(commaPos+1):-1])
            unique_test_movie_ids.add(testMovieID)
            row_data.append(testMovieID)
            for otherMovieID in row[1:]:
                otherMovieID = int(otherMovieID)
                row_data.append(otherMovieID)
                unique_test_movie_ids.add(otherMovieID)
            raw_test_data.append(row_data)
        
        return np.array(raw_test_data)
        
def read_csv_data(filename):
    return pd.read_csv(os.path.join(DATA_DIR, filename), header=None).to_numpy()
    
def read_georg(filename):
    matrix = pd.read_csv(os.path.join(DATA_DIR, filename), header=None).to_numpy()
    for row in matrix:
        row[1:] = (rankdata(row[1:]) - 1).astype(int)
    return matrix
    
    
        
        
def get_merged_ratings(list_of_rankings):
    merged_rankings = np.zeros(list_of_rankings[0].shape)
    for ranking in list_of_rankings:
        merged_rankings[:,1:] += ranking[:,1:]
    
    for user_row in merged_rankings:
        user_row /= user_row.max()
    
    merged_rankings[:,0] = list_of_rankings[0][:,0]
    
    return merged_rankings
    
    
            
            
gmf_results = read_baseline_data("gmf.test.prediction")

mlp_results = read_baseline_data("mlp.test.prediction")

neumf_results = read_baseline_data("neumf.test.prediction")

vae_results = read_csv_data("vae.csv")

ngcf_results = read_csv_data("ngcf.csv")

#attention_results = read_georg("george.txt")



#print(attention_results)

merged_rankings = get_merged_ratings([gmf_results, mlp_results, neumf_results, vae_results, ngcf_results])

num_of_users = merged_rankings.shape[0]
num_of_movies = merged_rankings.shape[1] - 1

list_of_positive_samples = [0] * num_of_users

list_of_negative_samples = [list(range(1,num_of_movies)) for i in range(num_of_users)]

predicition_matrix = merged_rankings[:,1:]
#print(list_of_positive_samples)
#print(list_of_negative_samples)

for i in range(1,11):
    print(i)
    hr_dist, ndcg_dist = eval_ratings_custom_at_end(predicition_matrix, list_of_positive_samples, list_of_negative_samples, i)

    #hr_dist, ndcg_dist = eval_ratings_custom_at_end(vae_results[:,1:], list_of_positive_samples, list_of_negative_samples)

    print("hr: " + str(hr_dist.mean()))
    print("ndcg: " + str(ndcg_dist.mean()))