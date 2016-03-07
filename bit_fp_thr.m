function e_val = bit_fp_thr(bits, no_labels)

    cdf = zeros(1,length(0:bits));
    for d=0:bits
        ball_size = sum(arrayfun(@nchoosek,bits*ones(1,d+1),0:d));
        out = 1;
        for i=0:no_labels-1
            out = out*max(0,1-(ball_size*i)/(2^bits));
        end
        cdf(d+1) = out;
        %fprintf('dist. d: %d, ball_size:%d, prob. %.3f\n', ...
        %    d,ball_size, out);       
    end
    
    e_val = sum(diff(fliplr(cdf)) .* [bits:-1:1]);
end
