% Q2
clear;
close all;

load face.mat
% 520 images of size 56x46

%% Partition
% 80% for training + 20% for testing
indexXtrain = 1:1:8;
indexXtest = 9:10;

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

% sustract mean face
averageFace = repmat(averageFace, [1,520]);
A = X - averageFace;

% covariance matrix (1/N)AT*A
S = A.'*A/520;

% eigenvector of S
[eigVector, eigValue] = eig(S);
eigValue = diag(eigValue);
[eigValueSort, sortID] = sort(eigValue,'descend');
eigFaces = eigVector(:,sortID(1:50));

% visualize eigenFaces
figure
for iEigenFaces = 1:1:50
    eigFaceU = A*eigFaces(:,iEigenFaces);
    eigFaceDisplay = reshape(eigFaceU,56,46);
    subplot(5,10,iEigenFaces);
    imagesc(eigFaceDisplay),colormap('gray');
end
