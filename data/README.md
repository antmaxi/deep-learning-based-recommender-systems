# Datasets, Preprocessing, and Format.

## General Information about the Selected Datasets.

### MovieLens 1M ( ml-1m.zip )

MovieLens 1M movie ratings. Stable benchmark dataset. 1 million ratings from 6000 users on 4000 movies. Released 2/2003. 

https://grouplens.org/datasets/movielens/

All ratings are contained in the file "ratings.dat" and are in the
following format:

UserID::MovieID::Rating::Timestamp

- UserIDs range between 1 and 6040. 
- MovieIDs range between 1 and 3952.
- Ratings are made on a 5-star scale (whole-star ratings only).
- Timestamp is represented in seconds since the epoch as returned by time(2).
- Each user has at least 20 ratings.

### Jester ( jester_dataset_1_3.zip )

Data from 24,938 users who have rated between 15 and 35 jokes, a matrix with dimensions 24,938 X 101.

http://eigentaste.berkeley.edu/dataset/

Format:
- Data files are in .zip format, when unzipped, they are in Excel (.xls) format.
- Ratings are real values ranging from -10.00 to +10.00 (the value "99" corresponds to "null" = "not rated").
- One row per user.
- The first column gives the number of jokes rated by that user. The next 100 columns give the ratings for jokes 01 - 100.
- The sub-matrix including only columns {5, 7, 8, 13, 15, 16, 17, 18, 19, 20} is dense. Almost all users have rated those jokes.

### Epinions ( epinions (66mb) )

http://cseweb.ucsd.edu/~jmcauley/datasets.html#social_data

## Dataset Preprocessing.

For Movielens and Epinions, for each user we kept the most recent rated item (i.e. item corresponding to rating with max timestamp) as positive item in the test set.  
For Jester, we do not have timestamps. For each rating we generated a random timestamp, and then proceeded as in Movielens and Epinions.  
These are the positive items.  

Then we generate m negatives to test the ranking of the positive item against.
- For Movielens and Epinions we use m = 99.
- For Jester we use m = 49 (since there are only 100 items in the dataset).

Remark for the dataset filtering.
- All user and item IDs were made continuous so as to avoid the cold start problem, i.e. we removed users and items with no ratings at all.
- User ids and item ids are all zero-based.
- Duplicate samples were removed, i.e. cases where the same user rated the same item more than once.

## Data Format.

train.rating: 
- Train file.
- Each Line is a training instance: userID\t itemID\t rating\t timestamp

test.rating:
- Test file (positive instances). 
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp

test.negative
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing m negative samples.
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...
- For the case of Movielens and Epinions we set m = 99. For Jester m = 49.
