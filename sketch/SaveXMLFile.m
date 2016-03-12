function SaveXMLFile(setupStr, xmlFilePath)

docNode = com.mathworks.xml.XMLUtils.createDocument('Setup');
docRootNode = docNode.getDocumentElement;

% add child nodes
AddElementStr(docNode, docRootNode, 'DatasetName', setupStr.datasetName);
AddElementStr(docNode, docRootNode, 'DataTrnType', setupStr.dataTrnType);
AddElementStr(docNode, docRootNode, 'ClassCnt', num2str(setupStr.clsCnt));
AddElementStr(docNode, docRootNode, 'BatchCnt', num2str(setupStr.batchCnt));
AddElementStr(docNode, docRootNode, 'InstFeatDimCnt', num2str(setupStr.instFeatDimCnt));
AddElementStr(docNode, docRootNode, 'InstCntInBatch', num2str(setupStr.instCntInBatch));
AddElementStr(docNode, docRootNode, 'EvaluationType', setupStr.evaluationType);

% write <docNode> to a *.xml file
xmlwrite(xmlFilePath, docNode);

end

function docRootNode = AddElementStr(docNode, docRootNode, elementName, elementStr)

thisElement = docNode.createElement(elementName); 
thisElement.appendChild(docNode.createTextNode(elementStr));
docRootNode.appendChild(thisElement);

end