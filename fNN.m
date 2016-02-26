% Function for Nearest Neighbor
% MLCV CW1

% Input
% wTest = testing w vector [MxNtest]
% wTrain = training w matrix [MxNtrain]

% Output
% errorMinID = minimum error ID in training set

function [errorMinID, en] = fNN(wTest, wTrain)

    wTest = wTest.';
    wTrain = wTrain.';
    en = pdist2(wTrain,wTest);
    [~, errorMinID] = min(en);
    
end