# Deep Learning based Recommender Systems.

In the `data/` folder all the preprocessed data splits that we generated can be found, along with a README containing information about the datasets we used, the data preprocessing, and the data formatting.
The scripts we wrote to analyze and generate those data splits can be found in the `methods/neural_collaborative_filtering/Preprocess/` directory.
The main file is the `prep.m`.
Those data splits were generated in the format required in the neural collaborative filtering code, and then ported to the formats of the other approaches, so as to make sure that all codes get as input the exact same preprocessed data splits.

In the `methods/` folder, all the modified codes for the selected methods are given.
In every subfolder, a separate README can be found with the main modifications in each code as well as the instructions of how to use it.

The authors' original repositories that our codes were based on, can be found at:
- Neural Collaborative Filtering ( https://github.com/hexiangnan/neural_collaborative_filtering ).
- Collaborative Memory Network ( https://github.com/tebesu/CollaborativeMemoryNetwork ).
- Neural Graph Collaborative Filtering ( https://github.com/xiangwang1223/neural_graph_collaborative_filtering ).
- Variational Autoencoder ( https://github.com/dawenl/vae_cf ).
