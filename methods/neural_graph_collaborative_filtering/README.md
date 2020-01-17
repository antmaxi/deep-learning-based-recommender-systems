# Neural Graph Collaborative Filtering
Neural Graph Collaborative Filtering (NGCF) is a new recommendation framework based on graph neural network, explicitly encoding the collaborative signal in the form of high-order connectivities in user-item bipartite graph by performing embedding propagation.

Used code from https://github.com/xiangwang1223/neural_graph_collaborative_filtering based on the paper

>Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua (2019). Neural Graph Collaborative Filtering, [Paper in ACM DL](https://dl.acm.org/citation.cfm?doid=3331184.3331267) or [Paper in arXiv](https://arxiv.org/abs/1905.08108). In SIGIR'19, Paris, France, July 21-25, 2019.

Author: Dr. Xiang Wang (xiangwang at u.nus.edu)

Modified by Anton Maksimov

## Main Modifications
- Calculation of NDCG and hit ratio were changed for unification with other algorithms tested: 
 in test for each user there is one true rating and 99 (for Movielens and Epinions) and 49 (for Jester) wrong rating among which position of true one is found and correspondent metric is calculated.
- For each problem the full trainset along with the testset are stored in the Data/ directory (it is processed by data_convert.ipynb data from neural_collaborative_filtering/Data folder).
- The code now also computes more detailed statistics, i.e. Hit Ratio and NDCG in the whole range from 1 to 10 (can be specified in parser.py). 
- Finally, an option for writing the predicted item rankings of each method into a file was added, so that the study of the methods in an ensemble learning context is also possible.

## Datasets and Data Format.

Three new processed datasets are provided:
- MovieLens 1 Million (ml-1m).
- Jester 1.3 (jester).
- Epinions (epinions1).

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.8.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes (see the parser function in NGCF/utility/parser.py).
* Movielens dataset
```
python NGCF.py --dataset ml-1m --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
```

* Jester dataset
```
python NGCF.py --dataset jester --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
```

* Epinions dataset
 ```
python NGCF.py --dataset epinions1 --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
```

Some important arguments:

* `node_dropout`
  * It indicates the node dropout ratio, which randomly blocks a particular node and discard all its outgoing messages. Usage: `--node_dropout [0.1] --node_dropout_flag 1`
  * Note that the arguement `node_dropout_flag` also needs to be set as 1, since the node dropout could lead to higher computational cost compared to message dropout.

* `mess_dropout`
  * It indicates the message dropout ratio, which randomly drops out the outgoing messages. Usage `--mess_dropout [0.1,0.1,0.1]`.

## How to run on Leonhard
```
module load python_gpu/3.7.1
bsub -n 1 -W 10:00 -R 'rusage[mem=30720, ngpus_excl_p=1]' -J "output" python NGCF.py --dataset ml-1m --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
```