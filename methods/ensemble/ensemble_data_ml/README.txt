In this folder you will find the prediction data along with the respective
outputs of each specific run (which includes Hit Ratio and NDCG),
for the following architectures on the MOVIELENS dataset.

- Generalized Matrix Factorization
- Multi-Layer Perceptron
- Neural Matrix Factorization

For each one of them, the .txt has info and stats for the respective run, 
while the .prediction file has what you care about.

The format of the .prediction files is exactly like the .test.negative files in our datasets, 
with the only difference that the item ids are substituted by their predicted ranks [0, 99].
The 99 corresponds to the top predicted rank.
