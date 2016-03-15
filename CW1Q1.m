% Q1
clear;
close all;

load face.mat
% 520 images of size 56x46

%% Partition
% 80% for training + 20% for testing
indexX = randperm(10,10);
indexXtrain = indexX(1,1:8);
indexXtest = indexX(1,9:10);

Xtrain = [];
Xtest = [];

for iX = 0:51
    Xtrain = [Xtrain, X(:,indexXtrain+10*iX)];
    Xtest = [Xtest, X(:,indexXtest+10*iX)];
end

%% Training
% mean face
averageFace = mean(Xtrain,2);

% visualize mean face
aveFaceDisplay = reshape(averageFace, [56,46]);
figure
imshow(uint8(aveFaceDisplay),'Border', 'tight'),title('Mean face image');


% sustract mean face
averageFace = repmat(averageFace, [1,416]);
A = Xtrain - averageFace;

% covariance matrix (1/N)A*AT
S = A*A.'/416;
M = 50;

% eigenvector of S
[eigVector, eigValue] = eig(S);
eigValue = diag(eigValue);
[eigValueSort, sortID] = sort(eigValue,'descend');
eigFaces = eigVector(:,sortID(1:M));

% visualize first 16 eigenFaces
figure
for iEigenFaces = 1:16
    eigFaceDisplay = reshape(eigFaces(:,iEigenFaces)/norm(eigFaces(:,iEigenFaces)),[56,46]);
    subplot(4,4,iEigenFaces)
    imagesc(eigFaceDisplay),colormap('gray');
    axis 'off'
end
