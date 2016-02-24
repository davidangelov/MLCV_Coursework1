clear;
close all;

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

M = 50;
eigFaces = eigVector(:,sortID(1:M));
eigFacesU = A*eigFaces;

% normalization
for idU = 1:M
    
    eigFacesU(:,idU) = eigFacesU(:,idU)./norm(eigFacesU(:,idU));
    
end
%% Representing faces onto eigenfaces

% The columns of the matrix are projections
wMatrix = (A.'*eigFacesU).';


%% Reconstruct
XReconstruct = zeros(2576,416);

for id_n_recon = 1:1:416
    
    XReconstruct(:,id_n_recon) = averageFace(:,1) ...
        + eigFacesU*wMatrix(:,id_n_recon);
    
end


% reconstructed image
figure('Name','The first 20 reconstructed training faces')
for iX = 1:1:20
    
    subplot(4,5,iX)
    Xdisplay = reshape(XReconstruct(:,iX),[56,46]);
    imagesc(Xdisplay),colormap('gray');
    
end

% original training image
figure('Name','The first 20 training faces')
for iXtrain = 1:1:20
    subplot(4,5,iXtrain);
    Xdisplay = reshape(Xtrain(:,iXtrain),[56,46]);
    imagesc(Xdisplay),colormap('gray');
end

    
    
    
    