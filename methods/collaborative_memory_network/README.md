# Collaborative Memory Network for Recommendation Systems
Implementation for

**Authored by:** Travis Ebesu, Bin Shen, Yi Fang.: https://github.com/tebesu/CollaborativeMemoryNetwork.


**Modified by:** Georg R. Pollak.

## Modifications
- Fixed small bug in `util/layers.py` when trying to resume model training
``` python
if isinstance(self.config.optimizer_params, str):  
    self.config.optimizer_params = literal_eval(self.config.optimizer_params)
```
- Write output matrix to `ratings_matrix` file

### In order to pre-train on Leonhard
```
bsub -n 1 -R 'rusage[mem=<mem>, ngpus_excl_p=1]' python pretrain.py --gpu 0 --dataset <filepath> --output <where_to_save_pretrain_embedding> -e <embedding_size>
```
### In order to train on Leonhard
```
bsub -n 1 -R 'rusage[mem=<mem>, ngpus_excl_p=1]' python train.py --gpu 0 --logdir "log_dir" --dataset <filepath> --pretrain <pretrain_file> -e <embedding_size>
```
#### Requirements
* Python 3.6
* TensorFlow 1.4+
* dm-sonnet
