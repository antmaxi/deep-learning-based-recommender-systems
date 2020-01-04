# README 

# Datasets
### MovieLens 1M Dataset ( ml-1m.zip )
https://grouplens.org/datasets/movielens/

``` bash
userId,movieId,rating,timestamp
```
**Rating**: 5-star scale with 0.5 increments
TODO: add more info

### Jester Dataset ( jester-data-1.zip )
https://goldberg.berkeley.edu/jester-data/
4.1 Million continuous ratings (-10.00 to +10.00) of 100 jokes from 73,421 users: collected between April 1999 - May 2003.

- 3 Data files contain anonymous ratings data from 73,421 users.
- Data files are in .zip format, when unzipped, they are in Excel (.xls) format
- One row per user
- The first column gives the number of jokes rated by that user. The next 100 columns give the ratings for jokes 01 - 100.
- The sub-matrix including only columns {5, 7, 8, 13, 15, 16, 17, 18, 19, 20} is dense. 
  Almost all users have rated those jokes (see discussion of "universal queries" in the above paper).
- Ratings are real values ranging from -10.00 to +10.00 (the value "99" corresponds to "null" = "not rated").
  **Note**: for implicit 1: if between (0-10]
                         0: if 99 or in [-10,0] 

### Epinions ( epinions (66mb) )
http://cseweb.ucsd.edu/~jmcauley/datasets.html#social_data
These datasets include ratings as well as social (or trust) relationships between users. Data are from LibraryThing (a book review website) and epinions (general consumer reviews).


# Leonhard
### Uploading to the server 
Use rsync as it is super efficient

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
### Georg
Make Neural Memory Collaborative Filtering work with three datasets:
https://github.com/tebesu/CollaborativeMemoryNetwork
TODO fill out
### Nikolas
Do baseline with Neural Collaborative Filtering 
- https://github.com/hexiangnan/neural_collaborative_filtering
- https://github.com/yihong-chen/neural-collaborative-filtering
TODO fill out
### Anton
Make graph network 
https://github.com/xiangwang1223/neural_graph_collaborative_filtering
TODO fill out
### Anton
Philipe 
TODO fill out