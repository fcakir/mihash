function update_table = trigger_update(W, Xsample, Ysample, Hres, Hnew, reservoir_size, count)
typ = 6;
if (typ == 1)
    thr = 0.05;
    bitdiff = xor(Hres, Hnew);
    bf_temp = sum(bitdiff(:))/reservoir_size;
    update_table = bf_temp > thr;
    return;
elseif (typ == 2)
    % get 5% of the distances
    prop_neigh = 0.1;
    no_bits = size(Hres,2);
    assert(no_bits == size(Hnew,2));
    
    % get old threshold
    hdist = (2*Hres - 1) * (2*Hres - 1)';
    hdist = (-hdist + no_bits)./2;
    hdistU = hdist;
    hdistU(hdistU == 0) = -1;
    hdistU = triu(hdistU);
    hdistU(hdistU == 0) = [];
    hdistU(hdistU == -1) = 0;
    hdist_ = sort(hdistU,'ascend');
    N = histcounts(hdistU, no_bits-1, 'Normalization', 'probability');
    figure('Visible','off');
    bar(N);
    saveas(gcf, sprintf('/research/codebooks/hashing_project/data/old%05d.png', count));
    thr_dist = hdist_(ceil(prop_neigh*numel(hdist_)));
    
    member_res = hdist < thr_dist;
    
    hdistn = (2*Hnew - 1)* (2*Hnew - 1)';
    hdistn = (-hdistn + no_bits)./2;
    hdistnU = hdistn;
    hdistnU(hdistnU == 0) = -1;
    hdistnU = triu(hdistnU);
    hdistnU(hdistnU == 0) = [];
    hdistnU(hdistnU == -1) = 0;
    hdistn_ = sort(hdistnU,'ascend');
    N = histcounts(hdistnU, no_bits-1, 'Normalization', 'probability');
    figure('Visible','off');
    bar(N);
    saveas(gcf, sprintf('/research/codebooks/hashing_project/data/new%05d.png', count));
    thr_distn = hdistn_(ceil(prop_neigh*numel(hdistn_)));
    
    member_new = hdistn < thr_distn;
    
    ovl = sum(single(and(member_res, member_new)),2)./sum(single(or(member_res, member_new)),2);
    update_table = mean(ovl);
    %update_table = mean(ovl) > 0.1;

elseif (typ == 3)     
    no_bits = size(Hres,2);
    assert(no_bits == size(Hnew,2));
    
    % get old threshold
    hdist = (2*Hres - 1) * (2*Hres - 1)';
    hdist = (-hdist + no_bits)./2;
    
    [hdist_, hdistI_] = sort(hdist,'ascend');    
    
    hdistn = (2*Hnew - 1)* (2*Hnew - 1)';
    hdistn = (-hdistn + no_bits)./2;
    [hdistn_, hdistnI_] = sort(hdistn,'ascend');
    
    s = 0;
    for i=1: size(Hnew, 1)
        s = s + (1+corr(hdistI_(:,i), hdistnI_(:,i),'type','Kendall'))/2;
    end
       
    update_table = mean(s);
        
elseif typ == 4
    cateTrainTrain = (repmat(Ysample,1,length(Ysample)) == repmat(Ysample,1,length(Ysample))');
     % get 5% of the distances
    prop_neigh = 0.1;
    no_bits = size(Hres,2);
    assert(no_bits == size(Hnew,2));
    
    % get new 
    hdistn = (2*Hnew - 1)* (2*Hnew - 1)';
    hdistn = (-hdistn + no_bits)./2;
    A = hdistn(:); M = A(cateTrainTrain(:)); NM = A(~cateTrainTrain(:));
    %hdistnU = hdistn;
    %hdistnU(hdistnU == 0) = -1;
    %hdistnU = triu(hdistnU);
    %hdistnU(hdistnU == 0) = [];
    %hdistnU(hdistnU == -1) = 0;
    %hdistn_ = sort(hdistnU,'ascend');
    %figure('Visible','off');
    h1n = histcounts(M, 0:1:32, 'Normalization', 'probability');
    %figure('Visible','off');
    %bar(N);
    %hold on;
    h2n = histcounts(NM, 0:1:32, 'Normalization', 'probability');
    %bar(N);
    %%saveas(gcf, sprintf('/research/codebooks/hashing_project/data/misc/type4/new%05d.png', count));
    %thr_distn = hdistn_(ceil(prop_neigh*numel(hdistn_)));
    
    %member_new = hdistn < thr_distn;    
    
    % get old threshold
    hdist = (2*Hres - 1) * (2*Hres - 1)';
    hdist = (-hdist + no_bits)./2;
    A = hdist(:); M = A(cateTrainTrain(:)); NM = A(~cateTrainTrain(:));
    %hdistU = hdist;
    %hdistU(hdistU == 0) = -1;
    %hdistU = triu(hdistU);
    %hdistU(hdistU == 0) = [];
    %hdistU(hdistU == -1) = 0;
    %hdist_ = sort(hdistU,'ascend');
    %figure('Visible','off');
    h1 = histcounts(M, 0:1:32, 'Normalization', 'probability');
    %figure('Visible','off');
    %bar(N);
    h2 = histcounts(NM, 0:1:32, 'Normalization', 'probability');  
   
    hik = sum(min(h1, h2));
    tst = kstest2([h1n h2n], [h1 h2], 'Alpha', 0.001);
    figure('Visible','off');
    bar([h1' h2']);
    ylim([0 0.15]);
    legend(sprintf('HIK: %g KS Test: %d', hik, tst));
    saveas(gcf, sprintf('/research/codebooks/hashing_project/data/misc/type4-II/dist%05d.png', count));
    %thr_dist = hdist_(ceil(prop_neigh*numel(hdist_)));
    
    %member_res = hdist < thr_dist;
    
    
    %ovl = sum(single(and(member_res, member_new)),2)./sum(single(or(member_res, member_new)),2);
    %update_table = mean(ovl);
    update_table = -1;
elseif typ == 5
    cateTrainTrain = (repmat(Ysample,1,length(Ysample)) == repmat(Ysample,1,length(Ysample))');
     % get 5% of the distances
    no_bits = size(Hres,2);
    assert(no_bits == size(Hnew,2));
    
    % get new 
    hdist = (2*Hres - 1)* (2*Hres - 1)';
    hdist = (-hdist + no_bits)./2;
    shik  = 0;
    % make this faster
    for j=1:size(Hres,1)
        A = hdist(j, :); M = A(cateTrainTrain(j, :)); NM = A(~cateTrainTrain(j, :));
        h1 = histcounts(M, 0:1:32, 'Normalization', 'probability');
        h2 = histcounts(NM, 0:1:32, 'Normalization', 'probability');
        shik  = shik + sum(min(h1, h2));
    end
    
    shik = shik/reservoir_size;
 
    figure('Visible','off');
    bar([1-shik shik]);
    ylim([0 1]);
    legend(sprintf('Average separability: %g, non-separability: %g', 1-shik, shik));
    saveas(gcf, sprintf('/research/codebooks/hashing_project/data/misc/type4-III-5K/dist%05d.png', count));
    
    update_table = -1;
elseif (typ == 6)
    cateTrainTrain = (repmat(Ysample,1,length(Ysample)) == repmat(Ysample,1,length(Ysample))');
    no_bits = size(Hres,2);
    assert(isequal(no_bits, size(Hnew,2)));
    assert(isequal(reservoir_size,size(Hres,1), size(Hnew,1)));
    % if Q is the (hamming) distance - x axis
    % estimate P(Q|+), P(Q|-) & P(Q)
    hdist = (2*Hres - 1)* (2*Hres - 1)';
    hdist = (-hdist + no_bits)./2;   
    condent = zeros(1,reservoir_size);
    Qent = zeros(1, reservoir_size);
    % make this faster
    for j=1:reservoir_size
        A = hdist(j, :); M = A(cateTrainTrain(j, :)); NM = A(~cateTrainTrain(j, :));
        prob_Q_Cp = histcounts(M, 0:1:32, 'Normalization', 'probability'); % P(Q|+)
        prob_Q_Cn = histcounts(NM, 0:1:32, 'Normalization', 'probability'); % P(Q|-)
        prob_Q    = histcounts([M NM], 0:1:32, 'Normalization','probability'); % P(Q)        
        prob_Cp   = length(M)/(length(M) + length(NM)); %P(+)
        prob_Cn   = 1 - prob_Cp; % P(-)
        
        % estimate H(Q) entropy
        for q = 1:length(prob_Q)
            if prob_Q(q) == 0, lg = 0; else lg = log2(prob_Q(q)); end
            Qent(j) = Qent(j) - prob_Q(q) * lg;
        end
        
        % estimate H(Q|C)
        p = 0;
        for q=1:length(prob_Q_Cp)
            if prob_Q_Cp(q) == 0, lg = 0; else lg = log2(prob_Q_Cp(q)); end
            p = p - prob_Q_Cp(q) * lg;
        end
        n = 0;
        for q=1:length(prob_Q_Cn)
            if prob_Q_Cn(q) == 0, lg = 0; else lg = log2(prob_Q_Cn(q)); end
            n = n - prob_Q_Cn(q) * lg;
        end
        condent(j) = p * prob_Cp + n * prob_Cn;    
    end
    
    assert(all(Qent-condent >= 0));
    % estimate P(Q)
    hdistn = (2*Hnew - 1)* (2*Hnew - 1)';
    hdistn = (-hdistn + no_bits)./2;   
    condentn = zeros(1,reservoir_size);
    Qentn = zeros(1, reservoir_size);
    % make this faster
    for j=1:reservoir_size
        A = hdistn(j, :); M = A(cateTrainTrain(j, :)); NM = A(~cateTrainTrain(j, :));
        prob_Q_Cp = histcounts(M, 0:1:32, 'Normalization', 'probability'); % P(Q|+)
        prob_Q_Cn = histcounts(NM, 0:1:32, 'Normalization', 'probability'); % P(Q|-)
        prob_Q    = histcounts([M NM], 0:1:32, 'Normalization','probability'); % P(Q)        
        prob_Cp   = length(M)/(length(M) + length(NM)); %P(+)
        prob_Cn   = 1 - prob_Cp; % P(-)
        
        % estimate H(Q) entropy
        for q = 1:length(prob_Q)
            if prob_Q(q) == 0, lg = 0; else lg = log2(prob_Q(q)); end
            Qentn(j) = Qentn(j) - prob_Q(q) * lg;
        end
        
        % estimate H(Q|C)
        p = 0;
        for q=1:length(prob_Q_Cp)
            if prob_Q_Cp(q) == 0, lg = 0; else lg = log2(prob_Q_Cp(q)); end
            p = p - prob_Q_Cp(q) * lg;
        end
        n = 0;
        for q=1:length(prob_Q_Cn)
            if prob_Q_Cn(q) == 0, lg = 0; else lg = log2(prob_Q_Cn(q)); end
            n = n - prob_Q_Cn(q) * lg;
        end
        condentn(j) = p * prob_Cp + n * prob_Cn;    
    end
    
    assert(all(Qentn - condentn >= 0));
    
    figure('Visible','off');
    bar([mean(Qent) mean(Qent - condent) mean(Qentn - condentn)]);
    ylim([0 10]);
    legend(sprintf('Max mean MI :%g, Current mean MI: %g, New mean MI: %g', mean(Qent), mean(Qent - condent), mean(Qentn - condentn)));
    saveas(gcf, sprintf('/research/codebooks/hashing_project/data/misc/type6-I/ent%05d.png', count));
    update_table = sum(Qent - condent);
    
end


end

% 
%     t_ = tic;
%     M = zeros(reservoir_size);
%     NM = zeros(reservoir_size);
%     CTT = mat2cell(cateTrainTrain, ones(1, reservoir_size), reservoir_size);
%     M_ = hdist(cateTrainTrain); M(cateTrainTrain) = M_;
%     NM_ = hdist(~cateTrainTrain); NM(~cateTrainTrain) = NM_;
%     M = mat2cell(M, ones(1, reservoir_size), reservoir_size);
%     NM = mat2cell(NM, ones(1, reservoir_size), reservoir_size);
%     M = cellfun(@(x, y) x(y) , M, CTT, 'UniformOutput', 0);
%     NM = cellfun(@(x, y) x(~y) , NM, CTT, 'UniformOutput', 0);
%     
%     M = cellfun(@(x)histcounts(x, 0:1:32, 'Normalization','probability'), M, 'UniformOutput', 0);
%     NM = cellfun(@(x)histcounts(x, 0:1:32, 'Normalization','probability'), NM, 'UniformOutput', 0);
%     shik = cellfun(@(x, y) sum(min(x,y)), M, NM);
%     shik2 = sum(shik)/reservoir_size;
%     t1 = toc(t_);
%     fprintf('CF: %g (%g) FL:%g (%g) Difference: %g\n', shik2, shik, t1, t2, abs(shik - shik2));



