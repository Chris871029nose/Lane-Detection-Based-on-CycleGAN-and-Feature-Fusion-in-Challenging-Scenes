%% Calculate overall Fmeasure from each scenarios
clc; clear; close all;

allFile = 'output/ERFNet_iou0.5_split.txt';

all = textread(allFile,'%s');
TP = 0;
FP = 0;
FN = 0;

for i=1:9
   tpline = (i-1)*14+4;
   tp = str2double(all(tpline));
   fp = str2double(all(tpline+2));
   fn = str2double(all(tpline+4));
   TP = TP + tp;
   FP = FP + fp;
   FN = FN + fn;
end

P = TP/(TP + FP)
R = TP/(TP + FN)
F = 2*P*R/(P + R)*100
