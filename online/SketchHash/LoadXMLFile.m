function setupStr = LoadXMLFile(xmlFilePath)

xDoc = xmlread(xmlFilePath);

listItemLst = xDoc.getElementsByTagName('Setup');
assert(listItemLst.getLength == 1); % must have one and only one item
listItem = listItemLst.item(0); % select the only one list-item

% fetch configurations in the XML file
setupStr.datasetName = GetElementStr(listItem, 'DatasetName');
setupStr.dataTrnType = GetElementStr(listItem, 'DataTrnType');
setupStr.clsCnt = str2double(GetElementStr(listItem, 'ClassCnt'));
setupStr.batchCnt = str2double(GetElementStr(listItem, 'BatchCnt'));
setupStr.instFeatDimCnt = str2double(GetElementStr(listItem, 'InstFeatDimCnt'));
setupStr.instCntInBatch = str2double(GetElementStr(listItem, 'InstCntInBatch'));

% determine the evaluation positions
evaluationType = GetElementStr(listItem, 'EvaluationType');
if strcmp(evaluationType, 'All')
    setupStr.mapPosLst = (1 : setupStr.batchCnt);
elseif strcmp(evaluationType, 'Sel')
    setupStr.mapPosLst = [1, 2, 4, 8, 16, 32, 48, 64, 80, 100];
else
    error('invalid evaluation type: %s\n', evaluationType);
end

end

function elementStr = GetElementStr(listItem, elementName)

element = listItem.getElementsByTagName(elementName).item(0);
elementStr = char(element.getFirstChild.getData);

end