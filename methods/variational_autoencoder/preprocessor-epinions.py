#!/usr/bin/env python
# coding: utf-8

# # Variational autoencoders for collaborative filtering 

# This notebook accompanies the paper "*Variational autoencoders for collaborative filtering*" by Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara, in The Web Conference (aka WWW) 2018.
# 
# In this notebook, we will show a complete self-contained example of training a variational autoencoder (as well as a denoising autoencoder) with multinomial likelihood (described in the paper) on the public Movielens-20M dataset, including both data preprocessing and model training.

# In[2]:


import os
import shutil
import sys

import numpy as np
from scipy import sparse

import seaborn as sn
sn.set()

import pandas as pd

import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

import bottleneck as bn


# In[ ]:

# Config
DATA_DIR = '/Users/stduser/Documents/DeepLearningProject/data-project/'
pro_dir = os.path.join(DATA_DIR, 'pro_sg')
do_filter_triplets = False


# ## Data preprocessing

# We load the data and create train/validation/test splits following strong generalization: 
# 
# - We split all users into training/validation/test sets. 
# 
# - We train models using the entire click history of the training users. 
# 
# - To evaluate, we take part of the click history from held-out (validation and test) users to learn the necessary user-level representations for the model and then compute metrics by looking at how well the model ranks the rest of the unseen click history from the held-out users.

# First, download the dataset at http://files.grouplens.org/datasets/movielens/ml-20m.zip

# In[3]:


### change `DATA_DIR` to the location where movielens-20m dataset sits
#DATA_DIR = '/home/ubuntu/data/ml-20m/'
#DATA_DIR = 'C:/Users/Korba/Documents/Philippe2/ml-20m'



# In[4]:


#raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)
raw_data = pd.read_csv(os.path.join(DATA_DIR, 'epinions1.train.rating'), sep='\t')
raw_data.columns = ['userId','movieId','rating','timestamp']


# In[5]:


# binarize the data (only keep ratings >= 4)
#raw_data = raw_data[raw_data['rating'] > 3.5]


# In[6]:


raw_data['rating'] = (raw_data['rating'] > 0.0).astype(int)


# In[7]:


raw_data.head()


import csv

unique_test_movie_ids = set()
unique_test_user_ids = set()

raw_test_data = []

with open(os.path.join(DATA_DIR, 'epinions1.test.negative')) as csvfile:
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


# ### Data splitting procedure

# - Select 10K users as heldout users, 10K users as validation users, and the rest of the users for training
# - Use all the items from the training users as item set
# - For each of both validation and test user, subsample 80% as fold-in data and the rest for prediction 

# In[8]:


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


# In[9]:



def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users. 
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]
    
    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]
    
    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId') 
    return tp, usercount, itemcount

 


# Only keep items that are clicked on by at least 5 users

# In[10]:


if do_filter_triplets:
    raw_data, user_activity, item_popularity = filter_triplets(raw_data)
else:
    user_activity, item_popularity = get_count(raw_data, 'userId'), get_count(raw_data, 'movieId')


# In[11]:


sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" % 
      (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))


# In[12]:


unique_uid = user_activity.index


unique_user_ids = set()
unique_train_user_ids = set(unique_uid)
unique_user_ids.update(unique_train_user_ids)
unique_user_ids.update(unique_test_user_ids)

unique_uid = np.array(list(unique_user_ids))

np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size)
unique_uid = unique_uid[idx_perm]


# In[13]:


# In[14]:


# create train/validation/test users
n_users = unique_uid.size
n_heldout_users = int(n_users*0.1)

tr_users = unique_uid[:(n_users - n_heldout_users)]
vd_users = unique_uid[(n_users - n_heldout_users):]


# In[15]:


train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]


# In[16]:


unique_sid = pd.unique(train_plays['movieId'])

unique_movie_ids = set()
unique_train_movie_ids = set(unique_sid)
unique_movie_ids.update(unique_train_movie_ids)
unique_movie_ids.update(unique_test_movie_ids)

unique_sid = np.array(list(unique_movie_ids))


# In[17]:
"""

import pickle

with open('show2id.pickle', 'rb') as handle:
    show2id = pickle.load(handle)
    
#print(show2id[0])
    
with open('profile2id.pickle', 'rb') as handle:
    profile2id = pickle.load(handle)
    
#print(profile2id[0])
"""

show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

# In[18]:

if not os.path.exists(pro_dir):
    os.makedirs(pro_dir)

with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)


# In[19]:


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 100 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    
    return data_tr, data_te


# In[20]:


vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
print(vad_plays)
vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]


# In[21]:


print(vad_plays)


# In[22]:


vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)


# In[23]:


# ### Save the data into (user_index, item_index) format

# In[25]:


def numerize(tp):
    uid = list(map(lambda x: profile2id[x], tp['userId']))
    sid = list(map(lambda x: show2id[x], tp['movieId']))
    df = pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])
    return df
    

def numerize_alt(tp):
    uid = list(tp['userId'])
    sid = list(tp['movieId'])
    df = pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])
    return df
    


# In[26]:

print(train_plays)
train_data = numerize(train_plays)
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)


# In[27]:


vad_data_tr = numerize(vad_plays_tr)
vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)


# In[28]:


vad_data_te = numerize(vad_plays_te)
vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)


# In[29]:

"""
test_data_tr = numerize(test_plays_tr)
test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)


# In[30]:


test_data_te = numerize(test_plays_te)
test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
"""

#raw_test_data = pd.read_csv(os.path.join(DATA_DIR, 'ml-1m.test.negative'), sep='\t')

test_data_tr_custom = numerize(raw_data)
test_data_tr_custom.to_csv(os.path.join(pro_dir, 'test_data_tr_custom.csv'), index=False)

numerized_test_data = []

print(len(raw_test_data))

for row in raw_test_data:
    numerized_row_data = []
    numerized_user_id = profile2id[row[0]]
    numerized_row_data.append(numerized_user_id)
    for movieID in row[1:]:
        numerized_row_data.append(show2id[movieID])
        
    numerized_test_data.append(numerized_row_data)
    
    
        
df_test = pd.DataFrame.from_records(numerized_test_data)
df_test.to_csv(os.path.join(pro_dir, 'test_targets_and_negatives.csv'), index=False, header=False)