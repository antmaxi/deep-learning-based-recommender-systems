# README 

# Datasets
### MovieLens 1M Dataset ( ml-1m.zip )

MovieLens 1M movie ratings. Stable benchmark dataset. 1 million ratings from 6000 users on 4000 movies. Released 2/2003. 

https://grouplens.org/datasets/movielens/

All ratings are contained in the file "ratings.dat" and are in the
following format:

UserID::MovieID::Rating::Timestamp

- UserIDs range between 1 and 6040 
- MovieIDs range between 1 and 3952
- Ratings are made on a 5-star scale (whole-star ratings only)
- Timestamp is represented in seconds since the epoch as returned by time(2)
- Each user has at least 20 ratings

### Jester Dataset ( jester_dataset_1_3.zip )

Data from 24,938 users who have rated between 15 and 35 jokes, a matrix with dimensions 24,938 X 101.

http://eigentaste.berkeley.edu/dataset/

Format:
- Data files are in .zip format, when unzipped, they are in Excel (.xls) format
- Ratings are real values ranging from -10.00 to +10.00 (the value "99" corresponds to "null" = "not rated").
- One row per user.
- The first column gives the number of jokes rated by that user. The next 100 columns give the ratings for jokes 01 - 100.
- The sub-matrix including only columns {5, 7, 8, 13, 15, 16, 17, 18, 19, 20} is dense. Almost all users have rated those jokes (see discussion of "universal queries" in the above paper).

**Note**: for implicit 1: if between (0-10], and 0: if 99 or in [-10,0] 

### Epinions ( epinions (66mb) )
http://cseweb.ucsd.edu/~jmcauley/datasets.html#social_data

# Dataset Preprocessing

For Movielens and Epinions, for each user keep the most recent rated item (i.e. item corresponding to rating with max timestamp) as positive item in the test set.

For Jester, we do not have timestamps. For each user keep in the test set the rated item that has the max id.

These are the positive items.

Then generate k negatives to test the ranking of the positive item against.
For Movielens use k = 99.
For Jester use k = 49 (since there are only 100 items).
For Epinions lets try k = 99.

Remark for the filtering of the datasets.
- Make sure that user and item ids are continuous so as to avoid cold start problem.
- Remove duplicate samples, i.e. cases where the same user rated the same item more than once.

# Evaluation

Use HitRatio@10 and NDCG@10.

# Leonhard
### Uploading to the server 
Use rsync as it is super efficient

``` bash
rsync -Pav src/ username@hostname:/destionation
```

### Submitting Jobs
#### Important Options
bsub options:"  
-N: send email at job end"  
-R: memory usage&number of gpus"  
-W:hh:mm"  
-n: number of processors"  
-J: jobname"  

### Tensorflow 
https://scicomp.ethz.ch/wiki/Python_on_Leonhard#TensorFlow
### Python 
https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs#Python_and_GPUs
i.e.
```bash
bsub -n 1 -W 00:05 -R 'rusage[mem=2048, ngpus_excl_p=1]' -J "output" python my_script.py
```
### Use $HOME/.bash_profile to load stuff on start up
#### Example
```bash
##################################################
## DEEP LEARNING PROJECT
echo "-> Loading modules required for Deep Leaning Project (see .bash_progile)"
echo "-Loading module python_gpu/3.6.4 -> Tensorflow 1.7"
echo "Installing python requirements for CollaborativeMemoryNetwork:"
cat "/cluster/home/pollakg/project/code/methods/CollaborativeMemoryNetwork/requirements.txt"
pip install --user -r "/cluster/home/pollakg/project/code/methods/CollaborativeMemoryNetwork/requirements.txt"
echo "To run ColloborativeMemoryNetowrk use:"
echo "bsub -n 1 -W 00:05 -R 'rusage[mem=2048, ngpus_excl_p=1]' -J "output" python train.py --gpu 0 --dataset data/citeulike-a.npz --pretrain pretrain/citeulike-a_e50.npz"
echo "bsub options:"
echo "-N: send email at job end"
echo "-R: memory usage&number of gpus"
echo "-W:hh:mm"
echo "-n: number of processors"
echo "-J: jobname"
##################################################
```
## Current ToDo's
1. Make Data Container class that handles all the data uniformly for everyone
2. Make an evaluation class 
### Georg
Make Neural Memory Collaborative Filtering work with three datasets:
- https://github.com/tebesu/CollaborativeMemoryNetwork
TODO fill out
### Nikolas
Do baseline with Neural Collaborative Filtering 
- https://github.com/hexiangnan/neural_collaborative_filtering ( This is the code I am using )
- https://github.com/yihong-chen/neural-collaborative-filtering ( Nice code if you want to have a look )
TODO fill out
### Anton
Make graph network 
- https://github.com/xiangwang1223/neural_graph_collaborative_filtering
TODO fill out
### Philippe 
- https://github.com/dawenl/vae_cf
TODO fill out
