import numpy as np
from collections import defaultdict


class Dataset(object):
    # train_data = [[user id, item id], ...]
    # test_data = [userid: [pos_id, [neg_id1, neg_id2, ..., neg_id100]]]
    def __init__(self, filename):
        """
        Wraps dataset and produces batches for the model to consume

        :param filename: path to training data for npz file
        """
        self._data = np.load(filename)
        #self.train_data = self._data['train_data'][:, :2]
        self.train_data = self._data['train_data']
        self.test_data = self._data['test_data'].tolist()
        self._train_index = np.arange(len(self.train_data), dtype=np.uint)
        self._n_users, self._n_items = self.train_data.max(axis=0) + 1

        # Neighborhoods
        self.user_items = defaultdict(set)
        self.item_users = defaultdict(set)
        for u, i in self.train_data:
            self.user_items[u].add(i)
            self.item_users[i].add(u)
        # Get a list version so we do not need to perform type casting
        self.item_users_list = {k: list(v) for k, v in self.item_users.items()}
        # maximum number of users that rated an item i
        self._max_user_neighbors = max([len(x) for x in self.item_users.values()])
        self.user_items = dict(self.user_items)
        self.item_users = dict(self.item_users)

    @property
    def train_size(self):
        """
        :return: number of examples in training set
        :rtype: int
        """
        return len(self.train_data)

    @property
    def user_count(self):
        """
        Number of users in dataset
        """
        return self._n_users

    @property
    def item_count(self):
        """
        Number of items in dataset
        """
        return self._n_items

    def _sample_item(self):
        """
        Draw an item uniformly
        """
        return np.random.randint(0, self.item_count)

    def _sample_negative_item(self, user_id):
        """
        Uniformly sample a negative item
        """
        if user_id > self.user_count:
            raise ValueError("Trying to sample user id: {} > user count: {}".format(
                user_id, self.user_count))

        n = self._sample_item()
        positive_items = self.user_items[user_id]

        if len(positive_items) >= self.item_count:
            raise ValueError("The User has rated more items than possible %s / %s" % (
                len(positive_items), self.item_count))
        # Item that exists in our set of items:  item_users but has not been rated by this user
        while n in positive_items or n not in self.item_users:
            n = self._sample_item()
        return n

    def _generate_data(self, neg_count):
        idx = 0
        self._examples = np.zeros((self.train_size*neg_count, 3),
                                  dtype=np.uint32)
        self._examples[:, :] = 0
        for user_idx, item_idx in self.train_data:
            for _ in range(neg_count):
                neg_item_idx = self._sample_negative_item(user_idx)
                self._examples[idx, :] = [user_idx, item_idx, neg_item_idx]
                idx += 1

    def get_data(self, batch_size: int, neighborhood: bool, neg_count: int):
        """
        Batch data together as (user, item, negative item), pos_neighborhood,
        length of neighborhood, negative_neighborhood, length of negative neighborhood

        if neighborhood is False returns only user, item, negative_item so we
        can reuse this for non-neighborhood-based methods.

        :param batch_size: size of the batch
        :param neighborhood: return the neighborhood information or not
        :param neg_count: number of negative samples to uniformly draw per a pos
                          example
        :return: generator
        """
        # Allocate inputs
        batch = np.zeros((batch_size, 3), dtype=np.uint32)
        pos_neighbor = np.zeros((batch_size, self._max_user_neighbors), dtype=np.int32)
        pos_length = np.zeros(batch_size, dtype=np.int32)
        neg_neighbor = np.zeros((batch_size, self._max_user_neighbors), dtype=np.int32)
        neg_length = np.zeros(batch_size, dtype=np.int32)

        # Shuffle index
        np.random.shuffle(self._train_index)

        idx = 0
        for user_idx, item_idx in self.train_data[self._train_index]:
            # TODO: set positive values outside of for loop
            for _ in range(neg_count):
                neg_item_idx = self._sample_negative_item(user_idx)
                batch[idx, :] = [user_idx, item_idx, neg_item_idx]

                # Get neighborhood information
                if neighborhood:
                    if len(self.item_users.get(item_idx, [])) > 0:
                        pos_length[idx] = len(self.item_users[item_idx])
                        pos_neighbor[idx, :pos_length[idx]] = self.item_users_list[item_idx]
                    else:
                        # Length defaults to 1
                        pos_length[idx] = 1
                        pos_neighbor[idx, 0] = item_idx

                    if len(self.item_users.get(neg_item_idx, [])) > 0:
                        neg_length[idx] = len(self.item_users[neg_item_idx])
                        neg_neighbor[idx, :neg_length[idx]] = self.item_users_list[neg_item_idx]
                    else:
                        # Length defaults to 1
                        neg_length[idx] = 1
                        neg_neighbor[idx, 0] = neg_item_idx

                idx += 1
                # Yield batch if we filled queue
                if idx == batch_size:
                    if neighborhood:
                        max_length = max(neg_length.max(), pos_length.max())
                        yield batch, pos_neighbor[:, :max_length], pos_length, \
                            neg_neighbor[:, :max_length], neg_length
                        pos_length[:] = 1
                        neg_length[:] = 1
                    else:
                        yield batch
                    # Reset
                    idx = 0

        # Provide remainder
        if idx > 0:
            if neighborhood:
                max_length = max(neg_length[:idx].max(), pos_length[:idx].max())
                yield batch[:idx], pos_neighbor[:idx, :max_length], pos_length[:idx], \
                    neg_neighbor[:idx, :max_length], neg_length[:idx]
            else:
                yield batch[:idx]
