# Data Format

train.rating: 
- Train file.
- Each Line is a training instance: userID\t itemID\t rating\t timestamp

test.rating:
- Test file (positive instances). 
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp

test.negative
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing k negative samples.
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...
- For the case of Movielens and Epinions we set k = 99. For Jester k = 49.
