# Abstract
Existing models ignore a major class of CF models:
- Neighborhood Models
- Memory Based Approaches

#### Collaborative Filtering (CF)
establish relevance between users and items from past interactions (i.e.\ ratings, clicks, purchases) by the assumption that similar users will consume similar items.
There exist three main groups of CF:
#### Memory/neighbor based approaches 
##### I.e. KNN
###### Advantage
    Good at capturing strong local relations
###### Drawback
    Only look at K-nearest neighbors => ignore mass of majority of ratings

#### Latent Factor Models
##### Matrix Factorization
    project users and items into a common low dimensional space capturing latent relations
###### Advantage
    Capture the overall global structure of the user and item relationships 
###### Drawback
    Often ignore the presence of a few strong local associations
#### Hybrid Approaches

## Why
   Existing and well established deep learning models incorporate the latent factor models but ignore the integration of the neighborhood-approach 
   in a non-linear fashion
## Approach
    Fuse:
    - a memory based component to represent the neighborhood based component in order to capture high order complex relations between items and user
    - an attention mechanism with implicit feedback in order to infer the _user specific_ contribution from the community
### Validation
    On three different test sets