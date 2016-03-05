function mAP = get_mAP(cateTrainTest, Y, tY)
	% input: 
	%  Y  - (logical) training binary codes
	%  tY - (logical) testing binary codes
	% output:
	%  mAP - mean Average Precision

	sim = single(2*Y-1)'*single(2*tY-1);
	trainsize = length(Y);
	testsize  = length(tY);

	AP = zeros(1, testsize);
	parfor j = 1:testsize
		labels = 2*double(cateTrainTest(:,j))-1;
		[~, ~, info] = vl_pr(labels, double(sim(:,j)));
		AP(j) = info.ap;
	end
	AP = AP(~isnan(AP));  % for NUSWIDE

	mAP = mean(AP);
	myLogInfo(['mAP = ' num2str(mAP)]);
end
