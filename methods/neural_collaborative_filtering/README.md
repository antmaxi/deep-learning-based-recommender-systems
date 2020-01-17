# Neural Collaborative Filtering (NCF)

This is the implementation for the paper:

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering.](http://dl.acm.org/citation.cfm?id=3052569) In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

Three deep learning based collaborative filtering models are implemented:
- Generalized Matrix Factorization (GMF).
- Multi-Layer Perceptron (MLP).
- Neural Matrix Factorization (NeuMF).

These models are optimized using log loss with negative sampling, so as to target implicit feedback and ranking task. 

**Authored by:** Dr. Xiangnan He (Original repository: https://github.com/hexiangnan/neural_collaborative_filtering).

**Modified by:** Nikolas Tselepidis.

## Main Modifications

- The original code was extended so as to be able to handle train-validation-test splits instead of just train-test splits, in order to have a more fair evaluation.
In the current version, the model is tuned on the train-validation split.
After the model hyperparameters are chosen, the model is being trained on the full trainset (union of train and validation) and then it is evaluated on the unseen test data.
For each problem the full trainset along with the testset are stored in the `Data/` directory.
The train-validation data split for each problem, can be found in the `Data/valid/` directory. 
- Moreover, the code now also computes more detailed statistics, i.e. HR@k and NDCG@k in the whole range from 1 to `topK=10`. 
- Finally, an option for writing the predicted item rankings of each method into a file was added, so that the study of the methods in an ensemble learning context is also possible.

## Datasets and Data Format.

Three new processed datasets are provided:
- MovieLens 1 Million (ml-1m).
- Jester 1.3 (jester).
- Epinions (epinions1).

train.rating:
- Train file.
- Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)

test.rating:
- Test file (positive instances). 
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)

test.negative
- Test file (negative instances).
- Each line corresponds to a line of test.rating, containing m negative samples (for Movielens and Epinions m=99, for Jester m=49). 
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...

## Predicted Rankings File Format.

test.prediction
- Each line is in the format: (userID,rankOfPositiveItem)\t rankOfNegativeItem1\t rankOfNegativeItem2 ...

## Example to run the codes locally.

First set up the local environment using the requirements.txt file.

```
python -m virtualenv ncf
source ncf/bin/activate
pip install -r requirements.txt
```

Run GMF:
```
python GMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --regs [0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1
```

Run MLP:
```
python MLP.py --dataset ml-1m --epochs 20 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1
```

Run NeuMF:
```
python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1
```

* **Note**: If you are using `zsh` and get an error like `zsh: no matches found: [64,32,16,8]`, you should use `single quotation marks` for array parameters like `--layers '[64,32,16,8]'`.

## Example to run the codes on Euler.

```
module load new gcc/4.8.2 python/2.7.12
export KERAS_BACKEND=theano
sh submit.sh ml-1m
sh submit.sh jester
sh submit.sh epinions1
```
In the submit.sh script various configurations of the models are tested.
