function train_sketch(run_trial, opts)
	% online sketching hashing

	global Xtrain Ytrain

	train_time  = zeros(1, opts.ntrials);
	update_time = zeros(1, opts.ntrials);
	bit_flips   = zeros(1, opts.ntrials);
	opts.nbatches = ceil(opts.noTrainingPoints / opts.batchsize);
	parfor t = 1:opts.ntrials
		if run_trial(t) == 0
			myLogInfo('Trial %02d not required, skipped', t);
			continue;
		end
		myLogInfo('%s: %d trainPts, random trial %d', opts.identifier, opts.noTrainingPoints, t);

		prefix = sprintf('%s/trial%d', opts.expdir, t);
		if opts.onlyfinal
			test_batchInds = opts.nbatches;
		else
			test_batchInds = zeros(1, opts.ntests);
			test_batchInds(1) = 1;
			test_batchInds(end) = opts.nbatches;
			interval = round(opts.nbatches/(opts.ntests-1));
			for i = 1:opts.ntests-2
				ind = interval*i + randi([1 round(interval/3)]) - round(interval/6);
				test_batchInds(i+1) = ind;
			end
		end

		% do Online Sketching Hashing
		[train_time(t), update_time(t), bit_flips(t)] = online_sketching_hashing(...
			Xtrain, Ytrain, prefix, test_batchInds, t, opts);
	end

	myLogInfo('Training time (total): %.2f +/- %.2f', mean(train_time), std(train_time));
	if strcmp(opts.mapping, 'smooth')
		myLogInfo('      Bit flips (per): %.4g +/- %.4g', mean(bit_flips), std(bit_flips));
	end
end



function [train_time, update_time, bitflips] = online_sketching_hashing(...
		Xtrain, Ytrain, prefix, test_batchInds, trialNo, opts)

	%%%% KH: convert parameters from opts to internal ones
	kInstFeatDimCnt = size(Xtrain, 2);  % feature dim
	bits = opts.nbits;


	% initialize hashing functions
	%{
	hashProjMat = cell(numel(kLoopBitsLst), 1);
	for ind = 1 : numel(kLoopBitsLst)
		bits = kLoopBitsLst(ind);
		hashProjMat{ind} = rand(kInstFeatDimCnt, bits) - 0.5;
	end
	%}
	W = rand(kInstFeatDimCnt, bits) - 0.5;
	H = [];  % initial hash table

	%{
	% create a variable to record time consuming
	timeElpsStr = struct(...
		'loadBtchData', zeros(batchCnt, 1), ...
		'updtHashFunc', zeros(batchCnt, 1), ...
		'calcHashCode', zeros(batchCnt, 1));
	%}
	train_time = 0;
	update_time = 0;

	%%%% KH: for counting bitflips
	ntrain_all = size(Xtrain, 1);
	bitflips = 0;

	% run online hashing methods
	instCntSeen = 0;
	instFeatAvePre = zeros(1, kInstFeatDimCnt);
	instFeatSkc = [];

	if opts.noTrainingPoints > 0
		numUseToTrain = opts.noTrainingPoints;
	else
		numUseToTrain = size(Xtrain, 1);
	end
	batchCnt = opts.nbatches;
	batchsize = opts.batchsize; %ceil(numUseToTrain/batchCnt);
	myLogInfo('%d batches of size %d', batchCnt, batchsize);

	for batchInd = 1 : batchCnt
		%fprintf('update hashing function with the %d-th batch\n', batchInd);

		%%%%%%%%%% LOAD BATCH DATA - BELOW %%%%%%%%%%
		%{
		%tic;

		% create directory to store hash code and evaluation result
		codeDirCur = sprintf('%s/%03d', kCodeDir_OSH, batchInd);
		rsltDirCur = sprintf('%s/%03d', kRsltDir_OSH, batchInd);
		mkdir(codeDirCur);
		mkdir(rsltDirCur);

		% obtain training data in the batch
		if strcmp(kDataTrnType, 'Solid')
			instFeatInBatch = dbTrn{batchInd};
		elseif strcmp(kDataTrnType, 'Stream')
			instFeatInBatch = importdata(dbTrnPathLst{batchInd});
		end
		%} 

		ind = (batchInd-1)*batchsize + 1 : min(batchInd*batchsize, numUseToTrain);
		instFeatInBatch = Xtrain(ind, :);

		instCntInBatch = size(instFeatInBatch, 1);

		%timeElpsStr.loadBtchData(batchInd) = toc;
		%%%%%%%%%% LOAD BATCH DATA - ABOVE %%%%%%%%%%


		%%%%%%%%%% UPDATE HASHING FUNCTION - BELOW %%%%%%%%%%
		tic;

		% calculate current mean feature vector
		instFeatAveCur = mean(instFeatInBatch, 1);

		% sketech current training batch
		if batchInd == 1
			instFeatToSkc = bsxfun(@minus, instFeatInBatch, instFeatAveCur);
		else
			instFeatCmps = sqrt(instCntSeen * instCntInBatch / (instCntSeen + instCntInBatch)) * (instFeatAveCur - instFeatAvePre);
			instFeatToSkc = [bsxfun(@minus, instFeatInBatch, instFeatAveCur); instFeatCmps];
		end
		instFeatSkc = MatrixSketch_Incr(instFeatSkc, instFeatToSkc, opts.sketchsize);

		% update mean feature vector and instance counter
		instFeatAvePre = (instFeatAvePre * instCntSeen + instFeatAveCur * instCntInBatch) / (instCntSeen + instCntInBatch);
		instCntSeen = instCntSeen + instCntInBatch;

		% compute QR decomposition of the sketched matrix
		[q, r] = qr(instFeatSkc', 0);
		[u, ~, ~] = svd(r, 'econ');
		v = q * u;

		%for ind = 1 : numel(kLoopBitsLst)
		% obtain the length of hashing code
		%bits = kLoopBitsLst(ind);

		% obtain the original projection matrix
		hashProjMatOrg = v(:, 1 : bits);

		% use random rotation
		R = orth(randn(bits));

		% update hashing function
		%hashProjMat{ind} = single(hashProjMatOrg * R);
		W = hashProjMatOrg * R;
		%end

		%timeElpsStr.updtHashFunc(batchInd) = toc;
		train_time = train_time + toc;

		%%%%%%%%%% UPDATE HASHING FUNCTION - ABOVE %%%%%%%%%%



		%%%%%%%%%% COMPUTE HASHING CODE - BELOW %%%%%%%%%%
		tic;

		% compute centered query/database instances
		%instFeatQryCen = bsxfun(@minus, instFeatQry, instFeatAvePre);
		%instFeatDtbCen = bsxfun(@minus, instFeatDtb, instFeatAvePre);

		instFeatDtbCen = bsxfun(@minus, Xtrain, instFeatAvePre);

		%{
		% compute hash code for query/database subset
		for ind = 1 : numel(kLoopBitsLst)
			bits = kLoopBitsLst(ind);

			%codeQry = (instFeatQryCen * hashProjMat{ind} > 0);
			%codeDtb = (instFeatDtbCen * hashProjMat{ind} > 0);

			% save hash code for query/database subset
			codeQryPath = sprintf('%s/codeQry.%d.mat', codeDirCur, bits);
			codeDtbPath = sprintf('%s/codeDtb.%d.mat', codeDirCur, bits);
			save(codeQryPath, 'codeQry');
			save(codeDtbPath, 'codeDtb');
		end
		%}

		if ~opts.onlyfinal || (opts.onlyfinal && batchInd==batchCnt)
			Hnew = (instFeatDtbCen * W > 0)';
			if ~isempty(H)
				bitdiff = xor(H, Hnew);
				bitdiff = sum(bitdiff(:))/ntrain_all;
				bitflips = bitflips + bitdiff;
				myLogInfo('[T%02d] HT update @%d, bitdiff=%g', trialNo, batchInd, bitdiff);
			else
				myLogInfo('[T%02d] HT update @%d', trialNo, batchInd);
			end
			H = Hnew;
			update_time = update_time + toc;
		end

		%timeElpsStr.calcHashCode(batchInd) = toc;
		%%%%%%%%%% COMPUTE HASHING CODE - ABOVE %%%%%%%%%%


		%%%% save intermediate models
		if (~opts.onlyfinal && ismember(batchInd, test_batchInds)) || ...
				(opts.onlyfinal && batchInd==batchCnt)
			F = sprintf('%s_batch%d.mat', prefix, batchInd);
			save(F, 'W', 'H', 'bitflips', 'train_time', 'update_time');
			if ~opts.windows, unix(['chmod o-w ' F]); end  % matlab permission bug
			myLogInfo('[T%02d] batch%d/%d Func %.2fs, Table %.2fs #BF=%g', ...
				trialNo, batchInd, batchCnt, train_time, update_time, bitflips);
		end

	end
	%{
	save(kRsltTimePath_OSH, 'timeElpsStr');
	fprintf('average time for loading batch data: %f\n', mean(timeElpsStr.loadBtchData));
	fprintf('average time for updating hashing function: %f\n', mean(timeElpsStr.updtHashFunc));
	fprintf('average time for computing hashing code: %f\n', mean(timeElpsStr.calcHashCode));
	%}

	% save final model, etc
	F = [prefix '.mat'];
	save(F, 'W', 'H', 'bitflips', 'train_time', 'update_time', 'test_batchInds');
	if ~opts.windows, unix(['chmod o-w ' F]); end % matlab permission bug
	myLogInfo('[T%02d] Saved: %s\n', trialNo, F);

	%{
	% evaluate the final hash code
	codeDirFnl = sprintf('%s/%03d', kCodeDir_OSH, batchCnt);
	rsltDirFnl = sprintf('%s/%03d', kRsltDir_OSH, batchCnt);
	rsltStrFnl = Evaluate_Fast(instCntInClsDtb, instCntInClsQry, instLinkQryGt, kLoopBitsLst, codeDirFnl, rsltDirFnl);
	fprintf('MAP = %f\n', rsltStrFnl.MAPval);
	%}
end
