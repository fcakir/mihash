function [imdb, imdbName] = get_imdb(opts)
if ~isempty(strfind(opts.modelType, 'fc')) || opts.imageSize <= 0
    imdbName = sprintf('%s_fc7', opts.dataset);
else
    imdbName = opts.dataset;
end
imdbFunc = str2func(['IMDB.' imdbName]);

% complete imdbName
imdbName = sprintf('%s_split%d', imdbName, opts.split);
if opts.normalize
    imdbName = [imdbName '_normalized'];
end

% imdbFile
imdbFile = fullfile(opts.dataDir, ['imdb_' imdbName '.mat']);
logInfo(imdbFile);

% load/save
t0 = tic;
try
    imdb = load(imdbFile) ;
    logInfo('loaded in %.2fs', toc(t0));
catch
    imdb = imdbFunc(opts) ;
    save(imdbFile, '-struct', 'imdb', '-v7.3') ;
    logInfo('saved in %.2fs', toc(t0));
end

end
