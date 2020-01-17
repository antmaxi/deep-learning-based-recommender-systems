function data = load_epinions()

fid = fopen('../epinions_data/epinions.json');
i = 1;
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    disp([num2str(i) ' ' tline])
    
    sline = split(tline);
    user = sline(2); user = user{1}(2:end-2);
    rate = sline(4); rate = str2num(rate{1});
    time = sline(6); time = str2num(time{1}(1:end-1));
    item = sline(10); item = item{1}(2:end-2);
    
    users{i} = string( user );
    items{i} = string( item );
    stars{i} = rate;
    times{i} = time;
    
    i = i + 1;
end
fclose(fid);

users = vertcat( users{:} );
items = vertcat( items{:} );
stars = vertcat( stars{:} );
times = vertcat( times{:} );

unique_users = unique( users );
unique_items = unique( items );
nusers = length( unique_users );
nitems = length( unique_items );
[nusers nitems]
ndata = length(stars)

umap = containers.Map;
imap = containers.Map;
for i = 1 : nusers
    umap( char(unique_users(i)) ) = i;
end
for i = 1 : nitems
    imap( char(unique_items(i)) ) = i;
end

% Generate data
samples = zeros(ndata,4);
for i = 1 : ndata
    samples(i,1:4) = [ umap(char(users(i))) imap(char(items(i))) stars(i) times(i) ];
end

unique(stars)'

length(samples)
samples = unique(samples,'rows');
length(samples)

R = sparse( samples(:,1), samples(:,2), samples(:,3) );
T = sparse( samples(:,1), samples(:,2), samples(:,4) );

[I,J,V] = find(R);

unique(V)'

[I,J,V] = find(R > 5);

for z = 1 : length( I )
    sorted_samples = sortrows( samples( (samples(:,1) == I(z) & samples(:,2) == J(z)), : ), 4, 'descend');
    R(I(z),J(z)) = sorted_samples(1,3);
    T(I(z),J(z)) = sorted_samples(1,4);
end

% Assemble data
[USERS, ITEMS, RATINGS] = find(R);
[  ~  ,   ~  ,   TIMES] = find(T);
data = [USERS-1 ITEMS-1 RATINGS TIMES];
minmax( data(:,1:3)' )

% Remove users with very few ratings in dataset
ratings_per_user = full( sum( spones(R), 2 ) );
keep = ( ratings_per_user > 1 );
R = R(keep,:);
T = T(keep,:);
% Remove items with no ratings after update ( item_ids continuous )
ratings_per_item = full( sum( spones(R), 1 ) );
keep = ( ratings_per_item ~= 0 );
R = R(:,keep);
T = T(:,keep);

% Reassemble data
[ITEMS, USERS, RATINGS] = find(R');
[  ~  ,   ~  ,   TIMES] = find(T');
data = [USERS-1 ITEMS-1 RATINGS TIMES];
% data = [USERS-1 ITEMS-1 RATINGS randperm(length(TIMES))'];
minmax( data(:,1:3)' )

end