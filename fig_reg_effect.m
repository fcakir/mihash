% To run this script you need to initialize dataset to 'cifar' or 'places'
% dataset = 'cifar';
% dataset = 'places';

%----------------------CIFAR
if strcmp(dataset,'cifar')
    lambda = [0.01,0.05,0.1,0.25,0.5];

    refs = [];

    path='\\kraken\object_detection\cachedir\online-hashing\';
    colors = { [0.466 0.674 0.188], [0 0.447 0.741],[0.85,0.325,0.098], [0.929,0.694,0.125], [0.494, 0.184, 0.556]}; % , [0.301,0.745,0.933], [0.635, 0.078, 0.184],[0.95,0.95,0],[0.8,0,0.8]};

    h = figure;
    set(h,'position',[200 200 1000 600])
    hold on;

    for curve = 1:2
        result = [];
        if curve == 1 
            load([path 'cifar-cnn-32smooth-B0S0.1-U10\10000pts_10tests\mAP_5trials.mat']); 
            [it, avgmAP] = avg_curve(res, train_iter);
            avgBitFlips = mean(bitflips(:,end));
            area = (it(end)-it(1));
            avgmAP = avgmAP./area;
            baseline = (trapz(it, avgmAP))/avgBitFlips;
            refs = [refs baseline];
        elseif curve == 2 
            for i = 1:length(lambda)
                load([path 'cifar-cnn-32smooth-B0S0.1-RS50L' num2str(lambda(i)) 'U10\10000pts_10tests\mAP_5trials.mat']);
                [it, avgmAP] = avg_curve(res, train_iter);
                avgBitFlips = mean(bitflips(:,end));
                area = (it(end)-it(1));
                avgmAP = avgmAP./area;
                res = (trapz(it, avgmAP))/avgBitFlips;
                result = [result res];
            end
            p(curve) = plot(lambda, result, '-s', 'Color', colors{2}, 'LineWidth', 3, 'MarkerFaceColor', colors{2});
        end
    end

    for i = 1:1
        hline = refline(0,refs(i));
        set(hline,'LineStyle','--');
        set(hline,'LineWidth',3);
        set(hline,'Color',colors{2});
    end

    h=legend('Ours_{u=10} (\lambda^*)', 'Ours_{u=10} (\lambda=0)'); %'Ours\it{-dyn}', 
    set(h,'fontname','Helvetica', 'fontsize',20, 'TextColor',[.3 .3 .3], 'fontweight','bold');

    set(gca,'YGrid','on');
    set(gca,'gridlinestyle','-');
    set(gca,'XColor', [.3 .3 .3]);
    set(gca,'YColor', [.3 .3 .3]);
    set(gca,'fontsize', 18);
    set(gca,'fontweight', 'bold');

    x=xlabel('Regularization Parameter (\lambda)','fontname','Helvetica', 'fontsize',20, 'color',[.3 .3 .3], 'fontweight','bold');
    set(x, 'Units', 'Normalized');
    pos = get(x, 'Position');
    set(x, 'Position', pos - [0, 0.01, 0]);

    ylabel('AUC / #BitFlipsPerExample', 'fontname','Helvetica', 'fontsize',20, 'color',[.3 .3 .3], 'fontweight','bold');

%----------------------PLACES
elseif strcmp(dataset,'places')
    lambda = [0.01,0.05,0.1,0.25,0.5];

    refs = [];

    path='\\kraken\object_detection\cachedir\online-hashing\';
    colors = { [0.466 0.674 0.188], [0 0.447 0.741],[0.85,0.325,0.098], [0.929,0.694,0.125], [0.494, 0.184, 0.556]}; % , [0.301,0.745,0.933], [0.635, 0.078, 0.184],[0.95,0.95,0],[0.8,0,0.8]};

    h = figure;
    set(h,'position',[200 200 1000 600])
    hold on;

    for curve = 1:2
        result = [];
        if curve == 1 
            load([path 'placesL2500-cnn-32smooth-B0S0.1-U50\50000pts_10tests\mAP_3trials.mat']); 
            [it, avgmAP] = avg_curve(res, train_iter);
            avgBitFlips = mean(bitflips(:,end));
            area = (it(end)-it(1));
            avgmAP = avgmAP./area;
            baseline = (trapz(it, avgmAP))/avgBitFlips;
            refs = [refs baseline];
        elseif curve == 2 
            for i = 1:length(lambda)
                load([path 'placesL2500-cnn-32smooth-B0S0.1-RS50L' num2str(lambda(i)) 'U50\50000pts_10tests\mAP_3trials.mat']);
                [it, avgmAP] = avg_curve(res, train_iter);
                avgBitFlips = mean(bitflips(:,end));
                area = (it(end)-it(1));
                avgmAP = avgmAP./area;
                res = (trapz(it, avgmAP))/avgBitFlips;
                result = [result res];
            end
            p(curve) = plot(lambda, result, '-s', 'Color', colors{1}, 'LineWidth', 3, 'MarkerFaceColor', colors{1});
        end
    end


    for i = 1:length(refs)
        hline = refline(0,refs(i));
        set(hline,'LineStyle','--');
        set(hline,'LineWidth',3);
        set(hline,'Color',colors{i});
    end


    h=legend('Ours_{u=50} (\lambda^*)', 'Ours_{u=50} (\lambda=0)'); %'Ours\it{-dyn}', 
    set(h,'fontname','Helvetica', 'fontsize',20, 'TextColor',[.3 .3 .3], 'fontweight','bold');

    set(gca,'YGrid','on');
    set(gca,'gridlinestyle','-');
    set(gca,'XColor', [.3 .3 .3]);
    set(gca,'YColor', [.3 .3 .3]);
    set(gca,'fontsize', 18);
    set(gca,'fontweight', 'bold');

    x=xlabel('Regularization Parameter (\lambda)','fontname','Helvetica', 'fontsize',20, 'color',[.3 .3 .3], 'fontweight','bold');
    set(x, 'Units', 'Normalized');
    pos = get(x, 'Position');

    ylabel('AUC / #BitFlipsPerExample', 'fontname','Helvetica', 'fontsize',20, 'color',[.3 .3 .3], 'fontweight','bold');
end