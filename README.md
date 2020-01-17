# README 

## Neural Collaborative Filtering: 
Based on the work at:
- https://github.com/hexiangnan/neural_collaborative_filtering

Main configurations and parameters tested:
- Embedding Sizes: 8,16,32,64
- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 15
- Learning Rate: 0.001
- Regularization: L2(0)
- Batch Size: 256

## Collaborative Memory Network
Based on the work at:
- https://github.com/tebesu/CollaborativeMemoryNetwork

Main configurations and parameters tested:
- Embedding Sizes: 8,16,32,46
- Optimizer: RMSprop with decay=0.9, momentum=0.9, seems to be similar to adam
- Epochs: 30
- Learning Rate: 0.001
- Regularization: L2(0.1)
- Batch Size: 128

## Neural Graph Collaborative Filtering:
Based on the work at:
- https://github.com/xiangwang1223/neural_graph_collaborative_filtering

Main configurations and parameters tested:
- Embedding Sizes: 8,16,32,64
- Batch Size: 1024
- Learning Rate: 0.0005
- Epochs: 400
- Layer size: [64,64,64]
- Node dropout: 0.1
- Messages dropout: [0.1,0.1,0.1]
- Regularization: 1e-5

# Variational Autoencoder:
Based on the work at:
- https://github.com/dawenl/vae_cf
