#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import sys
import random
import math
import heapq

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sn
sn.set()

import pandas as pd

import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

import bottleneck as bn


# In[ ]:

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Config
DATA_DIR = '/Users/stduser/Documents/DeepLearningProject/data-project'
pro_dir = os.path.join(DATA_DIR, 'pro_sg')
metric_length = 50


# ## Model definition and training


class MultiDAE(object):
    def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, random_seed=None):
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        self.dims = self.q_dims + self.p_dims[1:]
        
        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed

        self.construct_placeholders()

    def construct_placeholders(self):        
        self.input_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.dims[0]])
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)

    def build_graph(self):

        self.construct_weights()

        saver, logits = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)

        # per-user average negative log-likelihood
        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * self.input_ph, axis=1))
        # apply regularization to weights
        reg = l2_regularizer(self.lam)
        reg_var = apply_regularization(reg, self.weights)
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        loss = neg_ll + 2 * reg_var
        
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll)
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        return saver, logits, loss, train_op, merged

    def forward_pass(self):
        # construct forward graph        
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights) - 1:
                h = tf.nn.tanh(h)
        return tf.train.Saver(), h

    def construct_weights(self):

        self.weights = []
        self.biases = []
        
        # define weights
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            weight_key = "weight_{}to{}".format(i, i+1)
            bias_key = "bias_{}".format(i+1)
            
            self.weights.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            
            self.biases.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            # add summary stats
            tf.summary.histogram(weight_key, self.weights[-1])
            tf.summary.histogram(bias_key, self.biases[-1])


# The objective of Multi-VAE^{PR} (evidence lower-bound, or ELBO) for a single user $u$ is:
# $$
# \mathcal{L}_u(\theta, \phi) = \mathbb{E}_{q_\phi(z_u | x_u)}[\log p_\theta(x_u | z_u)] - \beta \cdot KL(q_\phi(z_u | x_u) \| p(z_u))
# $$
# where $q_\phi$ is the approximating variational distribution (inference model). $\beta$ is the additional annealing parameter that we control. The objective of the entire dataset is the average over all the users. It can be trained almost the same as Multi-DAE, thanks to reparametrization trick. 

# In[32]:


class MultiVAE(MultiDAE):

    def construct_placeholders(self):
        super(MultiVAE, self).construct_placeholders()

        # placeholders with default values when scoring
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)
        
    def build_graph(self):
        self._construct_weights()

        saver, logits, KL = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * self.input_ph,
            axis=-1))
        # apply regularization to weights
        reg = l2_regularizer(self.lam)
        
        reg_var = apply_regularization(reg, self.weights_q + self.weights_p)
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_ELBO = neg_ll + self.anneal_ph * KL + 2 * reg_var
        
        train_op = tf.train.AdamOptimizer(self.lr).minimize(neg_ELBO)

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll)
        tf.summary.scalar('KL', KL)
        tf.summary.scalar('neg_ELBO_train', neg_ELBO)
        merged = tf.summary.merge_all()

        return saver, logits, neg_ELBO, train_op, merged
    
    def q_graph(self):
        mu_q, std_q, KL = None, None, None
        
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)
        
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q = h[:, :self.q_dims[-1]]
                logvar_q = h[:, self.q_dims[-1]:]

                std_q = tf.exp(0.5 * logvar_q)
                KL = tf.reduce_mean(tf.reduce_sum(
                        0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q**2 - 1), axis=1))
        return mu_q, std_q, KL

    def p_graph(self, z):
        h = z
        
        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights_p) - 1:
                h = tf.nn.tanh(h)
        return h

    def forward_pass(self):
        # q-network
        mu_q, std_q, KL = self.q_graph()
        epsilon = tf.random_normal(tf.shape(std_q))

        sampled_z = mu_q + self.is_training_ph *            epsilon * std_q

        # p-network
        logits = self.p_graph(sampled_z)
        
        return tf.train.Saver(), logits, KL

    def _construct_weights(self):
        self.weights_q, self.biases_q = [], []
        
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i+1)
            bias_key = "bias_q_{}".format(i+1)
            
            self.weights_q.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            
            self.biases_q.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            # add summary stats
            tf.summary.histogram(weight_key, self.weights_q[-1])
            tf.summary.histogram(bias_key, self.biases_q[-1])
            
        self.weights_p, self.biases_p = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i+1)
            bias_key = "bias_p_{}".format(i+1)
            self.weights_p.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            
            self.biases_p.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            # add summary stats
            tf.summary.histogram(weight_key, self.weights_p[-1])
            tf.summary.histogram(bias_key, self.biases_p[-1])


# ### Training/validation data, hyperparameters

# Load the pre-processed training and validation data

# In[33]:


unique_sid = list()
with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())

n_items = len(unique_sid)


# In[34]:


def load_train_data(csv_file):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
                             
    print(data.shape)
    return data


# In[35]:


train_data = load_train_data(os.path.join(pro_dir, 'train.csv'))


# In[36]:


def load_tr_te_data(csv_file_tr, csv_file_te):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    print("Start_idx:{}".format(start_idx))
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())
    print("End_idx:{}".format(end_idx))

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te
    
def gen_negative_samples(vad_data_tr, n=metric_length-1):
    row_length = vad_data_tr.shape[1]
    vad_data_tr = vad_data_tr.todense()
    sample_lists = []
    for user_row in vad_data_tr:
        nonzero_entries = user_row.nonzero()[1]
        #print(nonzero_entries)
        #nonzero_entries = nonzero_entries.reverse()
        sampling_list = list(range(0,row_length))
        for nonzero_entry in nonzero_entries:
            sampling_list.remove(nonzero_entry)
        
        samples = random.sample(sampling_list,n)
        sample_lists.append(samples)
        
    return sample_lists
    
def gen_postive_test_lists(vad_data_te):
    vad_data_te = vad_data_te.todense()
    test_lists = []
    for user_row in vad_data_te:
        nonzero_entries = user_row.nonzero()[1]
        test_lists.append(list(nonzero_entries))
        
    return test_lists
        
        
         


# In[37]:


vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'),
                                           os.path.join(pro_dir, 'validation_te.csv'))
                                           
vad_data_negatives = gen_negative_samples(vad_data_tr)
vad_data_postive_test = gen_postive_test_lists(vad_data_te)


# Set up training hyperparameters

# In[38]:


N = train_data.shape[0]
idxlist = list(range(N))

# training batch size
batch_size = 100 # paper: 500
batches_per_epoch = int(np.ceil(float(N) / batch_size))

N_vad = vad_data_tr.shape[0]
idxlist_vad = range(N_vad)

# validation batch size (since the entire validation set might not fit into GPU memory)
batch_size_vad = 200000000

# the total number of gradient updates for annealing
total_anneal_steps = 200000
# largest annealing parameter
anneal_cap = 0.2


# Evaluate function: Normalized discounted cumulative gain (NDCG@k) and Recall@k

# In[39]:


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


# In[40]:


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)

    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall
    
    
def eval_ratings_custom(predicted_preference_matrix, k=10):
    hr_dist = []
    ndcg_dist = []
    for positive_test_samples, negative_test_samples, pred_user_preferences in zip(vad_data_postive_test, vad_data_negatives, predicted_preference_matrix):
        #print(positive_test_samples)
        if len(positive_test_samples) <= 0:
            continue
        if len(negative_test_samples) <= 0:
            continue
        target_test_sample = random.sample(positive_test_samples, 1)[0]
        #print(target_test_sample)
        #print(target_test_sample)
        hr, ndcg = eval_one_rating(target_test_sample, negative_test_samples, pred_user_preferences)
        hr_dist.append(hr)
        ndcg_dist.append(ndcg)
        
    return np.array(hr_dist), np.array(ndcg_dist)
    
    
def eval_ratings_custom_at_end(predicted_preference_matrix, positive_test_sample_list, negative_test_samples_list, k=10):
    hr_dist = []
    ndcg_dist = []
    for positive_test_sample, negative_test_samples, pred_user_preferences in zip(positive_test_sample_list, negative_test_samples_list, predicted_preference_matrix):
        #print(positive_test_samples)
        target_test_sample = positive_test_sample
        #print(target_test_sample)
        hr, ndcg = eval_one_rating(target_test_sample, negative_test_samples, pred_user_preferences, k=k)
        hr_dist.append(hr)
        ndcg_dist.append(ndcg)
        
    return np.array(hr_dist), np.array(ndcg_dist)
        

def eval_ratings_custom_at_end_with_rankings_saving(predicted_preference_matrix, positive_test_sample_list, negative_test_samples_list, k=10):
    hr_dist = []
    ndcg_dist = []
    i = 0
    ranking_matrix = np.zeros((predicted_preference_matrix.shape[0], metric_length + 1))
    for positive_test_sample, negative_test_samples, pred_user_preferences in zip(positive_test_sample_list, negative_test_samples_list, predicted_preference_matrix):
        
        #print(positive_test_samples)
        target_test_sample = positive_test_sample
        
        ranking_matrix[i][0] = i
        ranking_matrix[i][1] = pred_user_preferences[target_test_sample]
        ranking_matrix[i][2:] = pred_user_preferences[negative_test_samples]
        
        ranking_matrix[i][1:] = (rankdata(ranking_matrix[i][1:]) - 1).astype(int)
        
        #print(target_test_sample)
        hr, ndcg = eval_one_rating(target_test_sample, negative_test_samples, pred_user_preferences)
        hr2, ndcg2 = eval_one_rating(0, list(range(1,metric_length)), ranking_matrix[i][1:])
        hr_dist.append(hr)
        ndcg_dist.append(ndcg)
        i += 1
        
    ranking_df = pd.DataFrame(ranking_matrix.astype(int))
    ranking_df.to_csv('ranking_matrix.csv', index=False, header=False)
        
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




# In[41]:


p_dims = [200, 600, n_items]


# In[42]:


tf.reset_default_graph()
vae = MultiVAE(p_dims, lam=0.0, random_seed=98765)

saver, logits_var, loss_var, train_op_var, merged_var = vae.build_graph()

ndcg_var = tf.Variable(0.0)
ndcg_dist_var = tf.placeholder(dtype=tf.float64, shape=None)
ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary])



# In[43]:


arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))


# In[44]:


log_dir = 'log/VAE_anneal{}K_cap{:1.1E}/{}'.format(
    total_anneal_steps/1000, anneal_cap, arch_str)

if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

print("log directory: %s" % log_dir)
summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())


# In[45]:


chkpt_dir = 'log/chkpt/VAE_anneal{}K_cap{:1.1E}/{}'.format(
    total_anneal_steps/1000, anneal_cap, arch_str)

if not os.path.isdir(chkpt_dir):
    os.makedirs(chkpt_dir) 
    
print("chkpt directory: %s" % chkpt_dir)


# In[46]:


n_epochs = 20


# In[47]:


ndcgs_vad = []

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

#with tf.Session(config=config) as sess:
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    best_ndcg = -np.inf

    update_count = 0.0

    
    for epoch in range(n_epochs):
        print("Starting epoch: " + str(epoch))

        np.random.shuffle(idxlist)
        # train for one epoch
        for bnum, st_idx in enumerate(range(0, N, batch_size)):
            end_idx = min(st_idx + batch_size, N)
            X = train_data[idxlist[st_idx:end_idx]]
            
            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')           
            
            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
            else:
                anneal = anneal_cap
            
            feed_dict = {vae.input_ph: X, 
                         vae.keep_prob_ph: 0.5, 
                         vae.anneal_ph: anneal,
                         vae.is_training_ph: 1}        
            sess.run(train_op_var, feed_dict=feed_dict)

            if bnum % 100 == 0:
                summary_train = sess.run(merged_var, feed_dict=feed_dict)
                summary_writer.add_summary(summary_train, 
                                           global_step=epoch * batches_per_epoch + bnum) 
            
            update_count += 1
        
        # compute validation NDCG
        hr_dist_custom = []
        ndcg_dist_custom = []
        for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
            end_idx = min(st_idx + batch_size_vad, N_vad)
            X = vad_data_tr[idxlist_vad[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')
        
            pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X} )
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            #ndcg_dist.append(NDCG_binary_at_k_batch(pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))
            #recall_dist.append(Recall_at_k_batch(pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))
            
            hr_dist_custom_run, ndcg_dist_custom_run = eval_ratings_custom(predicted_preference_matrix=pred_val, k=10)
            hr_dist_custom.append(hr_dist_custom_run)
            ndcg_dist_custom.append(ndcg_dist_custom_run)
            
            
            
            
        
        ndcg_dist_custom = np.concatenate(ndcg_dist_custom)
        ndcg_ = ndcg_dist_custom.mean()
        print("ndcg-mean-custom: %.3f, " % ndcg_) 

        hr_dist_custom = np.concatenate(hr_dist_custom)    
        hr_ = hr_dist_custom.mean() 
        print("hr-mean: %.3f, " % hr_) 

        ndcgs_vad.append(ndcg_)
        merged_valid_val = sess.run(merged_valid, feed_dict={ndcg_var: ndcg_, ndcg_dist_var: ndcg_dist_custom})
        summary_writer.add_summary(merged_valid_val, epoch)

        # update the best model (if necessary)
        saver.save(sess, '{}/model'.format(chkpt_dir))
        """
        if ndcg_ > best_ndcg:
            saver.save(sess, '{}/model'.format(chkpt_dir))
            best_ndcg = ndcg_
        """


#tp_tr = pd.read_csv(csv_file_tr)
###


test_data_tr = load_train_data(os.path.join(pro_dir, 'test_data_tr_custom.csv'))
import csv
# In[49]:



def load_test_targets(path_to_csv, num_of_users):
    with open(path_to_csv) as csvfile:
        positive_test_sample_list = [None] * num_of_users
        negative_test_samples_list = [None] * num_of_users
        readCSV = csv.reader(csvfile)
        for row in readCSV:
            userID = int(row[0])
            positive_test_sample_list[userID] = int(row[1])
            negative_test_samples_list[userID] = [int(d) for d in row[2:]]
            
    return positive_test_sample_list, negative_test_samples_list
            
            
    





positive_test_sample_list, negative_test_samples_list = load_test_targets(os.path.join(pro_dir, 'test_targets_and_negatives.csv'), num_of_users= test_data_tr.shape[0])

# In[50]:


N_test = test_data_tr.shape[0]
idxlist_test = range(N_test)

batch_size_test = 10000000#2000


# In[51]:


tf.reset_default_graph()
vae = MultiVAE(p_dims, lam=0.0)
saver, logits_var, _, _, _ = vae.build_graph()    


# Load the best performing model on the validation set

# In[52]:


chkpt_dir = 'log/chkpt/VAE_anneal{}K_cap{:1.1E}/{}'.format(
    total_anneal_steps/1000, anneal_cap, arch_str)
print("chkpt directory: %s" % chkpt_dir)


# In[53]:


hr_dist_custom, ndcg_dist_custom = [], []

with tf.Session() as sess:
    saver.restore(sess, '{}/model'.format(chkpt_dir))

    for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
        end_idx = min(st_idx + batch_size_test, N_test)
        X = test_data_tr[idxlist_test[st_idx:end_idx]]

        if sparse.isspmatrix(X):
            X = X.toarray()
        X = X.astype('float32')

        pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
        # exclude examples from training and validation (if any)
        pred_val[X.nonzero()] = -np.inf
        hr_dist_custom, ndcg_dist_custom = eval_ratings_custom_at_end_with_rankings_saving(predicted_preference_matrix=pred_val, positive_test_sample_list=positive_test_sample_list, negative_test_samples_list=negative_test_samples_list, k=10)
        for i in range(11):
            print("k = {}".format(i))
            hr_dist_custom, ndcg_dist_custom = eval_ratings_custom_at_end(predicted_preference_matrix=pred_val, positive_test_sample_list=positive_test_sample_list, negative_test_samples_list=negative_test_samples_list, k=i)
            print("Hit Ratio=%.5f (%.5f)" % (np.mean(hr_dist_custom), np.std(hr_dist_custom) / np.sqrt(len(hr_dist_custom))))
            print("NDCG%.5f (%.5f)" % (np.mean(ndcg_dist_custom), np.std(ndcg_dist_custom) / np.sqrt(len(ndcg_dist_custom))))
            