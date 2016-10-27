function joined_str = strjoin_fe(in_strings, delimiter)
    assert(iscellstr(in_strings), 'strjoin:cellstr', '1st argument: cell string');
    assert(ischar(delimiter) && (isvector(delimiter) || isempty(delimiter)), ...
           'strjoin:string', '2nd argument: string');

    append_delimiter = @(in) [in delimiter];
    appended = cellfun(append_delimiter, in_strings(1:end-1), 'UniformOutput', false);
    joined_str = horzcat(appended{:}, in_strings{end});
end