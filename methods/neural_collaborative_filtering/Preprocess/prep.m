% Loads the raw dataset and splits it into training and testing data
clear; clc; close all; rng(0);

data = load_movielens(); k = 99;
% data = load_jester(); k = 49;
% data = load_epinions(); k = 99;

[train_data, test_data, test_neg] = split_data( data, k );

% Check how many users and items exist in the test set but not in the
% training set
train_data = vertcat( train_data{:} );
disp('Number of users in the test set that are not in the training set')
length( setdiff( unique(test_data(:,1)), unique(train_data(:,1)) ) )
disp('Number of items in the test set that are not in the training set')
length( setdiff( unique(test_data(:,2)), unique(train_data(:,2)) ) )

size(train_data)
size(test_data)
size(test_neg)

return

disp('Generating Files...')

tic

write_cols('datagen/ml-1m.train.rating', train_data);
write_cols('datagen/ml-1m.test.rating', test_data);
write_neg('datagen/ml-1m.test.negative', test_data, test_neg);

write_cols('datagen/jester.train.rating', train_data);
write_cols('datagen/jester.test.rating', test_data);
write_neg('datagen/jester.test.negative', test_data, test_neg);

write_cols('datagen/epinions1.train.rating', train_data);
write_cols('datagen/epinions1.test.rating', test_data);
write_neg('datagen/epinions1.test.negative', test_data, test_neg);

toc