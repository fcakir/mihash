function update_table = trigger_update(W, Xsample, Ysample, Hres, Hnew, reservoir_size)
typ = 2;
if typ == 1
    thr = 0.05;
    bitdiff = xor(Hres, Hnew);
    bf_temp = sum(bitdiff(:))/reservoir_size;
    update_table = bf_temp > thr;
    return
elseif typ == 2
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
    thr_distn = hdistn_(ceil(prop_neigh*numel(hdistn_)));
    
    member_new = hdistn < thr_distn;
    
    ovl = sum(single(and(member_res, member_new)),2)./sum(single(or(member_res, member_new)),2);
    update_table = mean(ovl);
    %update_table = mean(ovl) > 0.1;

elseif typ == 3
     
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
    
    
end

end