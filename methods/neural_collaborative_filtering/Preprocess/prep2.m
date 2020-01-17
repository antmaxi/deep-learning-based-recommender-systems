% Loads the generated dataset and splits it into training and validation data
clear; clc; close all; rng(0);

% data = dlmread('datagen/ml-1m.train.rating'); k = 99;
% data = dlmread('datagen/jester.train.rating'); k = 49;
data = dlmread('datagen/epinions1.train.rating'); k = 99;

% !!!Use this block ONLY when processing epinions!!!
nnz( diff(unique(data(:,1))) ~= 1 )
nnz( diff(unique(data(:,2))) ~= 1 )
R = sparse(data(:,1)+1, data(:,2)+1, data(:,3));
T = sparse(data(:,1)+1, data(:,2)+1, data(:,4));
keep_users = full( sum(spones(R),2) ) > 1;
R = R(keep_users, :);
T = T(keep_users, :);
keep_items = full( sum(spones(R),1) ) ~= 0;
R = R(:, keep_items);
T = T(:, keep_items);
[it, us, ra] = find( R' );
[~,  ~,  ti] = find( T' );
data = [us-1 it-1 ra randperm(length(ti))']; % random timestamps

% Check for continuous user and item indices
nnz( diff(unique(data(:,1))) ~= 1 )
nnz( diff(unique(data(:,2))) ~= 1 )

[train_data, test_data, test_neg] = split_data( data, k );

% Check how many users and items exist in the validation (test) set and not
% in the (splitted) training set
train_data = vertcat( train_data{:} );
disp('Number of users in the validation set that are not in the training set')
length( setdiff( unique(test_data(:,1)), unique(train_data(:,1)) ) )
disp('Number of items in the validation set that are not in the training set')
length( setdiff( unique(test_data(:,2)), unique(train_data(:,2)) ) )

size(train_data)
size(test_data)
size(test_neg)

% return

disp('Generating Files...')

tic

% write_cols('datagen/valid/ml-1m.train.rating', train_data);
% write_cols('datagen/valid/ml-1m.test.rating', test_data);
% write_neg('datagen/valid/ml-1m.test.negative', test_data, test_neg);

% write_cols('datagen/valid/jester.train.rating', train_data);
% write_cols('datagen/valid/jester.test.rating', test_data);
% write_neg('datagen/valid/jester.test.negative', test_data, test_neg);

write_cols('datagen/valid/epinions1.train.rating', train_data);
write_cols('datagen/valid/epinions1.test.rating', test_data);
write_neg('datagen/valid/epinions1.test.negative', test_data, test_neg);

toc