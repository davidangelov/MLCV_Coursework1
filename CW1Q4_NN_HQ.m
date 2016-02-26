% MLCV coursework 1
% Q4 HQ

clc;
clear;
close all;

load face.mat
% 520 images of size 56x46

%% Partition
% 80% for training + 20% for testing

indexX = randperm(10,10);
indexXtrain = indexX(1,1:8);
indexXtest = indexX(1,9:10);

% take random sets
Xtrain = [];
Xtest = [];

for iX = 0:1:51
    Xtrain = [Xtrain, X(:,indexXtrain+10*iX)];
    Xtest = [Xtest, X(:,indexXtest+10*iX)];
end

%% Training
% mean face
averageFace = mean(Xtrain,2);

% visualize mean face
aveFaceDisplay = reshape(averageFace, 56,46);
figure
imshow(uint8(aveFaceDisplay));
title('Averaged face');

% sustract mean face
averageFace = repmat(averageFace, [1,416]);
A = Xtrain - averageFace;

% covariance matrix (1/N)AT*A
S = A.'*A/416;

% eigenvector of S
[eigVector, eigValue] = eig(S);
eigValue = diag(eigValue);
[eigValueSort, sortID] = sort(eigValue,'descend');

M = 100;
eigFaces = eigVector(:,sortID(1:M));
eigFacesU = A*eigFaces;

% normalization
for idU = 1:M
    
    eigFacesU(:,idU) = eigFacesU(:,idU)./norm(eigFacesU(:,idU));
    
end
%% Representing faces onto eigenfaces

% The columns of the matrix are projections
wMatrix = (A.'*eigFacesU).';

%% Testing
phiTest = Xtest - averageFace(:,1:size(Xtest,2));

wTest = (phiTest.'*eigFacesU).';

[predictedTrainID, en] = fNN(wTest,wMatrix);
% convert position to ID
predictedID = ceil(predictedTrainID/8);

trueID = reshape(repmat(1:52,2,1),1,52*2);

x = find(predictedID == trueID);
correctRate = length(find(predictedID == trueID))/104;
display(correctRate, 'Rate of correct prediction');
