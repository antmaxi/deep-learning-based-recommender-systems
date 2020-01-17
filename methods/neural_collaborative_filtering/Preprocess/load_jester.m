function data = load_jester()

data = csvread('../jester/jester-data-3.csv');

[m,n] = size(data);

data = data(:,2:end); % first col is just sum of ratings per user

data = data(:);
% data(data == 99) = 0;
% data(data > 0) = 1;
% data(data <= 0 ) = 0;
data( data >= -10 & data <= 10 ) = 1;
data( data == 99 ) = 0;
data = sparse( reshape(data,m,n-1) );

[user_ids, item_ids, ratings] = find( data );

unique_user_ids = unique( user_ids );
unique_item_ids = unique( item_ids );

n_users = length( unique_user_ids );
n_items = length( unique_item_ids );

[ length( unique_user_ids )  max( unique_user_ids ) ]
[ length( unique_item_ids )  max( unique_item_ids ) ]

umap = zeros( max(unique_user_ids), 1 );
umap( unique_user_ids ) = 1 : n_users;

imap = zeros( max(unique_item_ids), 1 );
imap( unique_item_ids ) = 1 : n_items;


user_ids = umap( user_ids ) - 1;
item_ids = imap( item_ids ) - 1;

data = [user_ids item_ids ratings randperm(length(item_ids))']; % artificial timestamp

minmax(data')
all( diff( unique( user_ids ) ) == 1 )
all( diff( unique( item_ids ) ) == 1 )

% R = full(sparse( user_ids+1, item_ids+1, ratings ));
% bar( sort( sum( R, 2 ) ) )

end