function DeltCodeRslt(codeDir, rsltDir)

% remove and create code directory
if exist(codeDir, 'dir')
    rmdir(codeDir, 's');
end
mkdir(codeDir);

% remove and create result directory
if exist(rsltDir, 'dir')
    rmdir(rsltDir, 's');
end
mkdir(rsltDir);

end