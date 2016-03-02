function mAP = get_mAP(cateTrainTest, Y, tY)
	sim = Y'*tY;
	trainsize = size(sim, 1);
	testsize  = size(sim, 2);

	AP = zeros(1, testsize);
	parfor j = 1:testsize
		labels = 2*double(cateTrainTest(:,j))-1;
		[~, ~, info] = vl_pr(labels, sim(:,j));
	end
	AP = AP(~isnan(AP));  % for NUSWIDE

	mAP = mean(AP);
	myLogInfo(['mAP = ' num2str(mAP)]);
end
