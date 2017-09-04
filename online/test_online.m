function test_online(res_fn, trial_fn, res_exist, opts)
% Copyright (c) 2017, Fatih Cakir, Kun He, Saral Adel Bargal, Stan Sclaroff 
% All rights reserved.
% 
% If used for please cite the below paper:
%
% "MIHash: Online Hashing with Mutual Information", 
% Fatih Cakir*, Kun He*, Sarah Adel Bargal, Stan Sclaroff
% (* equal contribution)
% International Conference on Computer Vision (ICCV) 2017
% 
% Usage of code from authors not listed above might be subject
% to different licensing. Please check with the corresponding authors for
% information.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice, this
%    list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
% 
% The views and conclusions contained in the software and documentation are those
% of the authors and should not be interpreted as representing official policies,
% either expressed or implied, of the FreeBSD Project.
%
%------------------------------------------------------------------------------
% Computes the performance by loading and evaluating the "checkpoint" files
% saved during training. 
%
% INPUTS
% 	res_fn - (string)Path to final results file.
%   trial_fn - (cell) 	Cell array containing the paths to individual trial results
% 			files.
% res_exist   - (int)   Boolean vector indicating whether the trial results files needed
% 			to be computed. For example, if opts.override=0 and the
% 		   	corresponding trial result file is computed (from a previous
% 			but identical experiment), the testing is skipped for that
% 			trial.
%	opts  - (struct)Parameter structure.
% 
% OUTPUTS
%	none

global Xtest Ytest Xtrain Ytrain thr_dist

if isempty(Ytrain)
    Affinity = pdist2(Xtrain, Xtest, 'euclidean') <= thr_dist; %logical
elseif size(Ytrain, 2) > 1
    Affinity = (Ytrain * testY' > 0);
elseif size(Ytrain, 2) == 1
    Affinity = bsxfun(@eq, Ytrain, testY');
else
    error('Ytrain error in test.m');
end

clear res bits_computed_all
clear train_iter train_time train_examples

for t = 1:opts.ntrials
    if res_exist(t)
        logInfo('Trial %d: results exist', t);
        load(trial_fn{t});
    else
        clear t_res t_bits_computed_all
        clear t_train_iter t_train_time

        Tprefix = sprintf('%s/trial%d', opts.expdir, t);
        Tmodel = load(sprintf('%s.mat', Tprefix));

        % handle transformations to X
        if strcmp(opts.methodID, 'okh')
            % do kernel mapping for test data
            testX_t = exp(-0.5*sqdist(Xtest', Tmodel.Xanchor')/Tmodel.sigma^2)';
            testX_t = [testX_t; ones(1,size(testX_t,2))]';
        elseif strcmp(opts.methodID, 'sketch')
            % subtract mean
            testX_t = bsxfun(@minus, Xtest, Tmodel.instFeatAvePre);
        else
            % OSH, AdaptHash: nothing
            testX_t = Xtest;
        end

        for i = 1:length(Tmodel.test_iters)
            % determine whether to actually run test or not
            % if there's no HT update since last test, just copy results
            if i == 1
                runtest = true;
            else
                st = Tmodel.test_iters(i-1);
                ed = Tmodel.test_iters(i);
                runtest = any(Tmodel.update_iters>st & Tmodel.update_iters<=ed);
            end

            iter = Tmodel.test_iters(i);
            d = load(sprintf('%s_iter%d.mat', Tprefix, iter));
            fprintf('Trial %d, Checkpoint %5d/%d, ', t, iter*opts.batchSize, ...
                opts.noTrainingPoints*opts.epoch);

            if runtest
                % test hash table
                % NOTE: for intermediate iters, need to use W_lastupdate (not W!)
                %       to compute Htest, to make sure it's computed using the same
                %       hash mapping as Htrain.
                Htest  = (testX_t * d.W_lastupdate > 0)';
                Htrain = d.H;

                % evaluate
                t_res(i) = evaluate(Htrain, Htest, Ytrain, Ytest, opts, Affinity);
                t_bits_computed_all(i) = d.bits_computed_all;
            else
                t_res(i) = t_res(i-1);
                t_bits_computed_all(i) = t_bits_computed_all(i-1);
                fprintf(' %g\n', t_res(i));
            end
            t_train_time(i) = d.train_time;
            t_train_iter(i) = iter;
        end
        clear Htrain Htest
        save(trial_fn{t}, 't_res', 't_bits_computed_all', ...
            't_train_iter', 't_train_time');
    end
    res(t, :) = t_res;
    bits_computed_all(t, :) = t_bits_computed_all;
    train_time(t, :) = t_train_time;
    train_iter(t, :) = t_train_iter;
    train_examples(t, :) = t_train_iter * opts.batchSize;
end
logInfo('  FINAL %s: %.3g +/- %.3g', opts.metric, mean(res(:,end)), std(res(:,end)));
logInfo('    AUC %s: %.3g +/- %.3g', opts.metric, mean(mean(res, 2)), std(mean(res, 2)));

% save all trials in a single file
save(res_fn, 'res', 'bits_computed_all', 'train_iter', 'train_time', ...
    'train_examples');

% visualize
if opts.showplots
    % draw curves, with auto figure saving
    figname = sprintf('%s_trex.fig', res_fn);
    show_res(figname, res, train_examples, 'Training Examples', opts.identifier, opts.override);
    figname = sprintf('%s_cpu.fig', res_fn);
    show_res(figname, res, train_time, 'CPU Time', opts.identifier, opts.override);
    figname = sprintf('%s_recomp.fig', res_fn);
    show_res(figname, res, bits_computed_all, 'Bit Recomputations', opts.identifier, opts.override);
    drawnow;
end
end

% -----------------------------------------------------------
function show_res(figname, Y, X, xlb, ttl, override)
try 
    assert(~override);
    openfig(figname);
catch
    [px, py] = avg_curve(Y, X);
    figure, if length(px) == 1, plot(px, py, '+'), else plot(px, py), end
    grid, title(ttl), xlabel(xlb), ylabel('res')
    saveas(gcf, figname);
    if isempty(strfind(computer, 'WIN'))
        unix(['chmod g+w ', figname]); 
        unix(['chmod o-w ', figname]); 
    end
end
end
