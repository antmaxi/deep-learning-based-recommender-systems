import numpy as np
import scipy.stats as ss

# Writes prediction matrix in file with format similar to .test.negative
def write_prediction_file(model, testRatings, testNegatives, method, dataset):
   
    path = 'Results/' + dataset + '/'
    f = open(path + method + '.test.prediction', 'w')
    
    # loop over all users
    for idx in xrange(len(testRatings)):
        # Extract data
        rating = testRatings[idx]  # [user index, positive item index]
        items = testNegatives[idx] # list containing test negatives for this user
        items.append(rating[1])    # append positive item to negative item list
        # Get prediction scores
        users = np.full(len(items), rating[0], dtype = 'int32')
        predictions = model.predict([users, np.array(items)], batch_size=100, verbose=0)
        # Transform scores to ranks
        ranks = map( np.int, ss.rankdata( predictions ) - 1 )
        user_idx  = rating[0]   # user index
        pos_item  = ranks[-1]   # positive item rank
        neg_items = ranks[0:-1] # negative item ranks
        # Write row in file
        f.write('(%d,%d)' % (user_idx, pos_item))
        for j in xrange(len(neg_items)):
            f.write('\t%d' % neg_items[j] )
        f.write('\n')

    f.close()
