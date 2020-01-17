function write_neg(filename, test_data, test_neg)

k = size(test_neg,2)

fid = fopen(filename, 'w');
for i = 1 : size( test_data, 1 )
    user = test_data(i,1);
    item = test_data(i,2);
    fprintf(fid, '(%d,%d)', user, item);
    for j = 1 : k
        fprintf(fid, '\t%d', test_neg(i,j));
    end
    fprintf(fid, '\n');
end
fclose(fid);

end