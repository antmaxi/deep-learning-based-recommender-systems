# README 

# Datasets

# Leonhard
## Submitting Jobs
### Important Options
bsub options:"  
-N: send email at job end"  
-R: memory usage&number of gpus"  
-W:hh:mm"  
-n: number of processors"  
-J: jobname"  
## Use $HOME/.bash_profile to load stuff on start up
##################################################
## Tensorflow 
https://scicomp.ethz.ch/wiki/Python_on_Leonhard#TensorFlow
## Python 
https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs#Python_and_GPUs
i.e.
'''bash
bsub -n 1 -W 00:05 -R 'rusage[mem=2048, ngpus_excl_p=1]' -J "output" python my_script.py
'''
### Example
# DEEP LEARNING PROJECT
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



# Current ToDo's
## Georg
Make Neural Memory Collaborative Filtering work with three datasets
## Nikolas
Do baseline with Neural Collaborative Filtering 
## Anton
Make some network 

# Papers