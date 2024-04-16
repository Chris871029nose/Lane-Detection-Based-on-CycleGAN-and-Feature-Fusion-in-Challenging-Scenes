%% Code to generate lane coordinates from probablity maps.
clc; clear; close all;

% Experiment name
exp = 'ERFNet';
% Data root
data = '/home/user/Datasets/culane';
% Directory where prob imgs generated by CNN are saved.
probRoot = strcat('../../predicts/', exp);
% Directory to save fitted lanes.
output = strcat('./output/', exp);

testList = strcat(data, '/list/test.txt');
show = false;  % set to true to visualize

list = textread(testList, '%s');
num = length(list);
pts = 18;

for i=1:num
    if mod(i,100) == 0
        fprintf('Processing the %d th image...\n', i);
    end
    imname = list{i};
    existPath = strcat(probRoot, imname(1:end-3), 'exist.txt');
    exist = textread(existPath, '%s');
    coordinates = zeros(4, pts);
    for j=1:4
        if exist{j}=='1'
            scorePath = strcat(probRoot, imname(1:end-4), '_', num2str(j), '_avg.png');
            scoreMap = imread(scorePath);
            coordinate = getLane(scoreMap);
            coordinates(j,:) = coordinate;
        end
    end
    if show
        img = imread(strcat(data, imname));
        probMaps = uint8(zeros(208,976,3));
        figure(1)
        imshow(img); hold on;
        for j=1:4
            color = ['g','b','r','y'];
            if exist{j}=='1'
                for m=1:pts
                    if coordinates(j,m)>0
                        plot(uint16(coordinates(j,m)*1640/976),590-(m-1)*20,strcat('.',color(j)),'markersize',30);
                    end
                end
            end
            probPath = strcat(probRoot, imname(1:end-4), '_', num2str(j), '_avg.png');
            probMap = imread(probPath);
            probMaps(:,:,mod(j,3)+1) = probMaps(:,:,mod(j,3)+1) + probMap;
        end
        hold off;
        figure(2)
        imshow(probMaps)
        pause();
    else
        save_name = strcat(output, imname(1:end-3), 'lines.txt');
        position = strfind(save_name,'/');
        prefix = '';
        if(~isempty(position))
            prefix = save_name(1:position(end));
        end
        if(~isdir(prefix) && ~strcmp(prefix,''))
            mkdir(prefix);
        end
        fp = fopen(save_name, 'wt');
        for j=1:4
            if exist{j}=='1' && sum(coordinates(j,:)>0)>1
                for m=1:pts
                    if coordinates(j,m)>0
                        fprintf(fp, '%d %d ', uint16(coordinates(j,m)*1640/976)-1, uint16(590-(m-1)*20)-1);
                    end
                end
                fprintf(fp, '\n');
            end
        end
        fclose(fp);
    end
end
