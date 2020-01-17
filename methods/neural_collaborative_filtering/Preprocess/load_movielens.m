function data = load_movielens()
%LOAD_MOVIELENS Loads movielens dataset.
% Returns data that is a nsamples x 4 matrix of the form
% [ user_id, item_id, rating, timestamp ]
% user_id column is a continuous range from 0 to n_users-1
% item_id column is a continuous range from 0 to n_users-1

% Load data
data = dlmread('../ml-1m/ratings.tab','\t');

% Unique user and item ids
user_ids = unique( data(:,1) );
item_ids = unique( data(:,2) );

% Number of users and items
n_users = length( user_ids );
n_items = length( item_ids );

[n_users max(user_ids)]
[n_items max(item_ids)]

% Map to transform item ids into continuous list
imap = zeros( max(item_ids), 1 );
imap( item_ids ) = 1 : n_items;

data(:,1) =      data(:,1)  - 1;
data(:,2) = imap(data(:,2)) - 1;

end