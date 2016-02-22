% MLCV coursework 1
% Q4 HQ

clc;
clear;
close all;

load face.mat
% 520 images of size 56x46

%% Partition
% 80% for training + 20% for testing
indexXtrain = 1:1:8;
indexXtest = 9:10;

% take random sets

Xtrain = []; %training faces
Xtest = []; %testing faces

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

M = 50;
eigFaces = eigVector(:,sortID(1:M));

% visualize first M = 50 eigenFaces
eigFacesU = zeros(2576,M);

figure
for iEigenFaces = 1:1:M
    eigFacesU(:,iEigenFaces) = A*eigFaces(:,iEigenFaces);
    eigFaceDisplay = reshape(eigFacesU(:,iEigenFaces),56,46);
    subplot(5,10,iEigenFaces);
    imagesc(eigFaceDisplay),colormap('gray');
end

%% Representing faces onto eigenfaces

% The columns of the matrix are projections
wMatrix = zeros(M,416);

for in = 1:1:416
    
    wMatrix(:,in) = (A(:,in).'*eigFacesU).';
    
end

%% Testing
% idX = 28;
idX = 3; % problem with this particular face?

testFace = Xtest(:,idX);
phiTest = testFace - averageFace(:,1);

wTest = phiTest.'*eigFacesU;
wTest = wTest.';

en = zeros(1,416);
for ien = 1:1:416
    en(ien) = norm(wTest-wMatrix(:,ien));
end
% pdist2

[~,idRecog] = min(en);
recogFace = Xtrain(:,idRecog);

figure
subplot(121)
testFace = reshape(testFace,56,46);
imagesc(testFace);colormap('gray');
title('Test face');
subplot(122)
recogFace = reshape(recogFace,56,46);
imagesc(recogFace);colormap('gray');
title('Recogised face');