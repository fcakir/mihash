% To run this script you need to initialize dataset to 'cifar' or 'places'
% dataset = 'cifar';
% dataset = 'places';

%----------------------CIFAR
if strcmp(dataset,'cifar')
    lambda = [0.01,0.05,0.1,0.25,0.5];
    update_interval = [10 100 1000];

    path='\\kraken\object_detection\cachedir\online-hashing\';
    colors = {[0.85,0.325,0.098], [.5 .5 .5], [0.494, 0.184, 0.556], [0.929,0.694,0.125], [0 0.447 0.741],[0.466 0.674 0.188]}; % , [0.301,0.745,0.933], [0.635, 0.078, 0.184],[0.95,0.95,0],[0.8,0,0.8]};

    h = figure;
    set(h,'position',[200 200 1000 700]);
    hold on;

    for curve = 1:6
        if curve == 1 
            load([path 'adapt\cifar-cnn-32smooth-A0.9B0.01S0.25-U50-50000pts-20tests\mAP_6trials.mat']);
        elseif curve == 2
            load([path 'cifar-cnn-32smooth-B1S0.1-U50\50000pts_20tests\mAP_5trials.mat']);
        elseif curve == 3
            load([path 'okh\cifar-cnn-32smooth-C1A0.1-U50-50000pts-20tests\mAP_6trials.mat']);
        elseif curve == 4
            load([path 'sketch\cifar-cnn-32smooth-sketch200-50000pts-100batches\mAP_6trials.mat']);
        elseif curve == 5 
            load([path 'cifar-cnn-32smooth-B0S0.1-U10\10000pts_10tests\mAP_5trials.mat']);
        elseif curve ==6
            load([path 'cifar-cnn-32smooth-B0S0.1-RS50L0.25Ada\50000pts_20tests\mAP_5trials.mat']);
        end
            [it, avgmAP] = avg_curve(res, train_iter);
            [~, avgBitFlips] = avg_curve(bitflips, train_iter);
            p(curve) = plot(log(avgBitFlips), avgmAP, '-s', 'Color', colors{curve}, 'LineWidth', 2, 'MarkerFaceColor', colors{curve},'MarkerSize',3);
    end

    h=legend('AdaptHash','OECC','OKH','SketchHash','Ours','Ours\it{-dyn}');
    set(h,'fontname','Helvetica', 'fontsize',20, 'TextColor',[.3 .3 .3], 'fontweight','bold');

    set(gca,'YGrid','on');
    set(gca,'gridlinestyle','-');
    set(gca,'XColor', [.3 .3 .3]);
    set(gca,'YColor', [.3 .3 .3]);
    set(gca,'fontsize', 16);
    set(gca,'fontweight', 'bold');

    x=xlabel('#BitFlipsPerExample (log scale)','fontname','Helvetica', 'fontsize',20, 'color',[.3 .3 .3], 'fontweight','bold');
    set(x, 'Units', 'Normalized');
    pos = get(x, 'Position');
    set(x, 'Position', pos - [0, 0.02, 0]);

    ylabel('mAP', 'fontname','Helvetica', 'fontsize',20, 'color',[.3 .3 .3], 'fontweight','bold');

%------------------ PLACES
elseif strcmp(dataset,'places')
    lambda = [0.01,0.05,0.1,0.25,0.5];
    update_interval = [10 100 1000];

    path='\\kraken\object_detection\cachedir\online-hashing\';
    colors = {[0.85,0.325,0.098], [.5 .5 .5], [0.494, 0.184, 0.556], [0.929,0.694,0.125], [0 0.447 0.741],[0.466 0.674 0.188]}; % , [0.301,0.745,0.933], [0.635, 0.078, 0.184],[0.95,0.95,0],[0.8,0,0.8]};

    h = figure;
    set(h,'position',[200 200 1000 700]);
    hold on;

    for curve = 1:6
        if curve == 1 
            load([path 'adapt\places-cnn-32smooth-A0.75B0.01S0.1-U50-100000pts-20tests\prec_n3_3trials.mat']);
        elseif curve == 2
            load([path 'places-cnn-32smooth-B1S0.1-U50\100000pts_20tests\prec_n3_3trials.mat']);
        elseif curve == 3
            load([path 'okh\places-cnn-32smooth-C1A0.1-U50-100000pts-20tests\prec_n3_3trials.mat']);
        elseif curve == 4
            load([path 'sketch\places-cnn-32smooth-sketch100-100000pts-100batches\prec_n3_3trials.mat']);
        elseif curve == 5
            load([path 'places-cnn-32smooth-B0S0.1-U50\100000pts_20tests\prec_n3_3trials.mat']);
        else
            load([path 'places-cnn-32smooth-B0S0.1-RS50L0.01Ada\100000pts_20tests\prec_n3_3trials.mat']);
        end
            [it, avgmAP] = avg_curve(res, train_iter);
            [~, avgBitFlips] = avg_curve(bitflips, train_iter);
            p(curve) = plot(log(avgBitFlips), avgmAP, '-s', 'Color', colors{curve}, 'LineWidth', 2, 'MarkerFaceColor', colors{curve},'MarkerSize',3);
    end


    h=legend('AdaptHash','OECC','OKH','SketchHash','Ours','Ours\it{-dyn}');
    set(h,'fontname','Helvetica', 'fontsize',20, 'TextColor',[.3 .3 .3], 'fontweight','bold');

    set(gca,'YGrid','on');
    set(gca,'gridlinestyle','-');
    set(gca,'XColor', [.3 .3 .3]);
    set(gca,'YColor', [.3 .3 .3]);
    set(gca,'fontsize', 16);
    set(gca,'fontweight', 'bold');
    set(gca,'Ylim',[0 0.31]);

    x=xlabel('#BitFlipsPerExample (log scale)','fontname','Helvetica', 'fontsize',20, 'color',[.3 .3 .3], 'fontweight','bold');
    set(x, 'Units', 'Normalized');
    pos = get(x, 'Position');
    set(x, 'Position', pos - [0, 0.02, 0]);

    ylabel('Precision@N=3', 'fontname','Helvetica', 'fontsize',18, 'color',[.3 .3 .3], 'fontweight','bold');
end
