function mAP = get_mAP(cateTrainTest, Y, tY)
	sim = Y'*tY;
	trainsize = size(sim, 1);
	testsize  = size(sim, 2);

	AP = zeros(1, testsize);
	parfor j = 1:testsize
		labels = 2*double(cateTrainTest(:,j))-1;
		[~, ~, info] = vl_pr(labels, sim(:,j));
		% temp fix for NUSWIDE -- why get NaN?
		if ~isnan(info.ap)
			AP(j) = info.ap;
		else
			AP(j) = info.ap_interp_11;
		end
	end

	mAP = mean(AP);
	myLogInfo(['mAP = ' num2str(mAP)]);
end
