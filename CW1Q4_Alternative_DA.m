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
% mean faces
MeanFaces = zeros(2576,52);

for iClasses = 0:51
    MeanFaces(:, iClasses + 1) = mean(Xtrain(:,(1:8)+iClasses*8),2);
end

% visualize mean face
% aveFaceDisplay = reshape(averageFace, 56,46);
% figure
% imshow(uint8(aveFaceDisplay));
% title('Averaged face');

% sustract mean face
averageFace = reshape(repmat(MeanFaces, [8,1]), 2576, 52*8);
A = Xtrain - averageFace;

% covariance matrix (1/N)AT*A
S = [];
for iClasses = 0:51
    AClass = A(:,(1:8)+iClasses*8);
    S = [S, AClass.'*AClass/416];
end


% eigenvector of S
eigVector = [];
eigValue = [];

for iClasses = 0:51
    SClass = S(:,(1:8)+iClasses*8);
    [eigVectorClass, eigValueClass] = eig(SClass);
    eigValueClass = diag(eigValueClass);
    eigVector = [eigVector, eigVectorClass];
    eigValue = [eigValue, eigValueClass];
end

[eigValueSort, sortID] = sort(eigValue,'descend');

M = 8;
eigFacesU = [];

for iClasses = 0:51
    eigFaces = eigVector(:,sortID(1:M, iClasses + 1));
    eigFacesU = [eigFacesU, A(:,(1:8)+iClasses*8)*eigFaces];
end

% normalization
for idU = 1:M*52
    
    eigFacesU(:,idU) = eigFacesU(:,idU)./norm(eigFacesU(:,idU));
    
end
%% Representing faces onto eigenfaces

% The columns of the matrix are projections
wTestMatrix = [];

%% Testing

averageFaceTest = reshape(repmat(MeanFaces, [2,1]), 2576, 52*2);
phiTest = Xtest - averageFaceTest;

for iClasses = 0:51
    TestClass = phiTest(:,(1:2)+iClasses*2);
    eigFacesUClass = eigFacesU(:,(1:M) + iClasses*M);
    wTestMatrix = [wTestMatrix, (TestClass.'*eigFacesUClass).'];
end

XReconstruct = [];

for iClasses = 0:51
    eigFacesUClass = eigFacesU(:,(1:M) + iClasses*M);
    wTestMatrixClass = wTestMatrix(:,(1:2)+iClasses*2);
    XReconstruct = [XReconstruct, eigFacesUClass*wTestMatrixClass];
end

XReconstruct = XReconstruct + averageFaceTest;

% averageFaceTest = reshape(repmat(MeanFaces, [2,1]), 2576, 52*2);
% phiTest = Xtest - averageFaceTest;
% 
% wTest = (phiTest.'*eigFacesU).';

% [predictedTrainID, en] = fNN(wTest,wMatrix);
% convert position to ID
% predictedID = ceil(predictedTrainID/8);
% 
% trueID = reshape(repmat(1:52,2,1),1,52*2);
% 
% x = find(predictedID == trueID);
% correctRate = length(find(predictedID == trueID))/104;
% display(correctRate, 'Rate of correct prediction');

Error_Array = pdist2(Xtest.',XReconstruct.');
[~,Min_error_Test_ID] = min(Error_Array);

predictedID = ceil(Min_error_Test_ID/2);

trueID = reshape(repmat(1:52,2,1),1,52*2);

x = find(predictedID == trueID);
correctRate = length(find(predictedID == trueID))/104;
display(correctRate, 'Rate of correct prediction');

% reconstructed testing image
% figure('Name','The first 20 reconstructed testing faces')
% for iX = 1:20
%     subplot(4,5,iX)
%     Xdisplay = reshape(XReconstruct(:,iX*2),[56,46]);
%     imagesc(Xdisplay),colormap('gray'); 
% end
% 
% % original testing image
% figure('Name','The first 20 testing faces')
% for iXtest = 1:20
%     subplot(4,5,iXtest);
%     Xdisplay = reshape(Xtest(:,iXtest*2),[56,46]);
%     imagesc(Xdisplay),colormap('gray');
% end
