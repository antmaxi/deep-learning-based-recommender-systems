function write_cols(filename, data)

fid = fopen(filename, 'w');
for i = 1 : length( data )
    user = data(i,1);
    item = data(i,2);
    rate = data(i,3);
    time = data(i,4);
    fprintf(fid, '%d\t%d\t%d\t%d\n', user, item, rate, time);
end
fclose(fid);

end