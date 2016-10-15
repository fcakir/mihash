function set_parpool(n)
p = gcp('nocreate');
if n <= 0
    if ~isempty(p)
        delete(p);
    end
elseif isempty(p)
    p = parpool(n);
elseif p.NumWorkers < n
    delete(p);
    p = parpool(n);
end
end
