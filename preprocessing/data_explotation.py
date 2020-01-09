import numpy as np
import pandas as pd
import random
from copy import deepcopy
import ast


class DataGenerator(object):
    """Construct dataset for deep learning project"""

    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        self.ratings["impl_rating"] = self.ratings.rating.apply(lambda x: 1 if (x > 0) else 0)
        # remove duplicates where a usr rated same item multiple times by using latest rating
        rows_to_keep = self.ratings.groupby(['userId', 'itemId']).timestamp.transform(max)
        self.ratings = self.ratings.loc[self.ratings.timestamp == rows_to_keep]
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        # create negative item samples for Mem learning
        self._sample_negative()
        breakpoint()
        new = self.ratings[["userId", "itemId", "timestamp", "impl_rating", "negative_samples"]].copy()

        self.train_ratings, self.test_ratings = self._train_test_split_loo(new)

    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""

        # ratings = ratings.copy()
        # ratings[ratings['rating'] > 0] = 1
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    def _train_test_split_loo(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        # Test sample are the ones with the heighest time stamp
        test = ratings[ratings['rank_latest'] == 1]
        # all others are in the traingis set
        train = ratings[ratings['rank_latest'] > 1]
        # Each user should at least have rated x samples => both sets should contain the same userIds
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'impl_rating']], test[['userId', 'itemId', 'impl_rating', 'negative_samples']]

    def random_sample(self, interacted, k):
        samples = set()
        i = k
        while len(samples) != k:
            s = random.sample(self.item_pool, i)
            s = set(s) - set(interacted)
            samples = samples | s
        return list(samples)

    def _sample_negative(self):
        """return all negative items & 100 sampled negative items"""
        # Creates for aach unique user a set of items that he interacted with
        interact_status = self.ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId': 'interacted_items'})
        self.ratings = pd.merge(self.ratings, interact_status, on='userId')
        # delet
        self.ratings = self.ratings.loc[self.ratings.interacted_items.apply(lambda x: len(x)) > 1]
        # interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        #interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 30))
        interacted = self.ratings['interacted_items'].to_numpy()
        negative_samples = []
        for idx, i in enumerate(interacted):
            if idx % 1000 == 0:
                pass
                # print(idx)
            negative_samples.append(random.sample(self.item_pool-set(i), 99))

        self.ratings['negative_samples'] = negative_samples
        assert self.ratings.shape[0] == len(negative_samples)
        #self.ratings['negative_samples'] = self.ratings["interacted_items"].apply(lambda i: self.random_sample(i, 99))
        # return interact_status[['userId', 'negative_items', 'negative_samples']]

    def save(self, filename, format="CMN"):
        """
        CMN format required
        - train_data.npy
            [[user id, item id], ...]
        - test_data.npy
            {userid: (pos_id, [neg_id1, neg_id2, ..., neg_id100])}
        """
        if format == "CMN":
            train_data = self.train_ratings[["userId", "itemId"]].to_numpy()
            test_data = self.test_ratings[["userId", "itemId", "negative_samples"]]
            #test_data = test_data.apply(lambda r: {r["userId"]: (r['itemId'], r["negative_samples"])}, axis=1).to_numpy()
            test_data = dict([(i, (a, b)) for i, a, b in zip(test_data.userId, test_data.itemId, test_data.negative_samples)])
            np.savez(filename, train_data=train_data, test_data=test_data)

        elif format == "NCF":
            pass
    # def instance_a_train_loader(self, num_negatives, batch_size):
    #     """instance train loader for one training epoch"""
    #     users, items, ratings = [], [], []
    #     train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
    #     train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))
    #     for row in train_ratings.itertuples():
    #         users.append(int(row.userId))
    #         items.append(int(row.itemId))
    #         ratings.append(float(row.rating))
    #         for i in range(num_negatives):
    #             users.append(int(row.userId))
    #             items.append(int(row.negatives[i]))
    #             ratings.append(float(0))  # negative samples get 0 rating
    #     dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
    #                                     item_tensor=torch.LongTensor(items),
    #                                     target_tensor=torch.FloatTensor(ratings))
    #     return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # @property
    # def evaluate_data(self):
    #     """create evaluate data"""
    #     test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
    #     test_users, test_items, negative_users, negative_items = [], [], [], []
    #     for row in test_ratings.itertuples():
    #         test_users.append(int(row.userId))
    #         test_items.append(int(row.itemId))
    #         for i in range(len(row.negative_samples)):
    #             negative_users.append(int(row.userId))
    #             negative_items.append(int(row.negative_samples[i]))
    #     return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
    #             torch.LongTensor(negative_items)]


if __name__ == "__main__":
    # read file
    lines = []
    with open('../data/epinions/epinions_data/epinions.json', 'r') as f:
        for nb, line in enumerate(f):
            lines.append(ast.literal_eval(line))

    # TODO delete duplicates
    df = pd.DataFrame(lines)
    epi_rating = df[["user", "item", "stars", "time"]]
    epi_rating.rename(columns={'stars': 'rating', 'time': 'timestamp'}, inplace=True)
    # Reindex
    unique_user_id = epi_rating[['user']].drop_duplicates().reindex()  # Create df of unique users
    unique_user_id['userId'] = np.arange(len(unique_user_id))  # append userId [uid, userId] [starts from 0, starts from 1]
    # Merge based on same uid => add userId with corresponding fitting uid
    epi_rating = pd.merge(epi_rating, unique_user_id, on=['user'], how='left')
    unique_item_id = epi_rating[['item']].drop_duplicates()
    unique_item_id['itemId'] = np.arange(len(unique_item_id))
    epi_rating = pd.merge(epi_rating, unique_item_id, on=['item'], how='left')
    epi_rating = epi_rating[['userId', 'itemId', 'rating', 'timestamp']]

    df = DataGenerator(epi_rating)

    breakpoint()
    df.save("/home/pollakg/polybox/CSE/master/2nd_term/Deep Learning/project/project-git/data/ml-1m/epinions.npz")
