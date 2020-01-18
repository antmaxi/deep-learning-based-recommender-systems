function [train_data, test_data, test_neg] = split_data( data, k )

n_users = max( data(:,1) ) + 1
n_items = max( data(:,2) ) + 1

train_data = cell( n_users, 1 );
test_data  = zeros( n_users, 4 );
test_neg   = zeros( n_users, k );

for user = 1 : n_users

    user_data = sortrows( data(data(:,1)==user-1,:), 4, 'descend' );

    train_data{ user } = user_data(2:end,:);

    test_data( user, : ) = user_data(1,:);

    item_vec = setdiff( 0 : n_items-1, user_data(2:end,2) );

    test_neg( user, : ) = item_vec( randperm( length(item_vec), k ) );

end

end