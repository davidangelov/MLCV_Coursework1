% Q3
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

for iX = 0:1:51
    Xtrain = [Xtrain, X(:,indexXtrain+10*iX)];
    Xtest = [Xtest, X(:,indexXtest+10*iX)];
end

%% Training
% mean face
averageFace = mean(Xtrain,2);

% visualize mean face
aveFaceDisplay = reshape(averageFace, 56,46);

% figure
% imshow(uint8(aveFaceDisplay));

% sustract mean face
averageFace = repmat(averageFace, [1,416]);
A = Xtrain - averageFace;

% covariance matrix (1/N)AT*A
S = A.'*A/416;

% number of basis to use
J_train_m = [];
J_train_theo_m = [];

M = 50;
% active this for loop to see J_theo vs J
% for M = 40:100

% eigenvector of S
[eigVector, eigValue] = eig(S);
eigValue = diag(eigValue);
[eigValueSort, sortID] = sort(eigValue,'descend');
eigFaces = eigVector(:,sortID(1:M));

eigFacesU = A*eigFaces;


for iEig = 1:M
    eigFacesU(:,iEig) = eigFacesU(:,iEig)./norm(eigFacesU(:,iEig));
end

% The columns of the matrix are projections
wMatrix = (A.'*eigFacesU).';

%% reconstruction of training image

A_U_recon = [];

for iX = 1:416
    A_U_sum = 0;
    for iEigU = 1:M
        A_U_sum = A_U_sum + wMatrix(iEigU,iX)*eigFacesU(:,iEigU);
    end
    A_U_recon = [A_U_recon, A_U_sum];
end

Xtrain_recon = averageFace + A_U_recon;

% figure
% imagesc(reshape(Xtrain_recon(:,1),56, 46));

J_train = sum(diag(pdist2(Xtrain.', Xtrain_recon.')).^2)/416;
J_train_theo = sum(eigValueSort(M+1:end));

J_train_m = [J_train_m, J_train];
J_train_theo_m = [J_train_theo_m, J_train_theo];

% end

%% J vs J_theo
figure(1),hold on
plot(40:M, J_train_m, 'bo-','MarkerSize', 10);
plot(40:M, J_train_theo_m, 'rx-','MarkerSize', 10);
xlabel('Number of eigenvectors used for PCA');
ylabel('Reconstruction error');
hold off, grid on
legend('Experimental error', 'Theorectical error');

%% visualizing reconstructed training image

figure(2),
subplot(321)
imagesc(reshape(Xtrain(:,1), 56,46)), axis 'off';
colormap('gray')
title('Face No. 1');

subplot(322)
imagesc(reshape(Xtrain_recon(:,1), 56,46)), axis 'off';
colormap('gray')
title('Reconstructed face No. 1');

subplot(323)
imagesc(reshape(Xtrain(:,10), 56,46)), axis 'off';
colormap('gray')
title('Face No. 2');

subplot(324)
imagesc(reshape(Xtrain_recon(:,10), 56,46)), axis 'off';
colormap('gray')
title('Reconstructed face No. 2');

subplot(325)
imagesc(reshape(Xtrain_recon(:,100), 56,46)), axis 'off';
colormap('gray')
title('Face No. 3');

subplot(326)
imagesc(reshape(Xtrain(:,100), 56,46)), axis 'off';
colormap('gray')
title('Reconstructed face No. 3');

%% reconstruction of testing image

phiTest = Xtest - averageFace(:,1:size(Xtest,2));
wTest = (phiTest.'*eigFacesU).';

A_te_recon = [];

for iX = 1:104
    A_U_sum = 0;
    for iEigU = 1:M
        A_U_sum = A_U_sum + wTest(iEigU,iX)*eigFacesU(:,iEigU);
    end
    A_te_recon = [A_te_recon, A_U_sum];
end

Xtest_recon = averageFace(:,1:size(Xtest,2)) + A_te_recon;
 
%% visulizing reconstructed testing images
figure(3),
subplot(321)
imagesc(reshape(Xtest(:,5), 56,46)), axis 'off';
colormap('gray')
title('Face No. 1');

subplot(322)
imagesc(reshape(Xtest(:,5), 56,46)), axis 'off';
colormap('gray')
title('Reconstructed face No. 1');

subplot(323)
imagesc(reshape(Xtest(:,10), 56,46)), axis 'off';
colormap('gray')
title('Face No. 2');

subplot(324)
imagesc(reshape(Xtest_recon(:,10), 56,46)), axis 'off';
colormap('gray')
title('Reconstructed face No. 2');

subplot(325)
imagesc(reshape(Xtest(:,100), 56,46)), axis 'off';
colormap('gray')
title('Face No. 3');

subplot(326)
imagesc(reshape(Xtest_recon(:,100), 56,46)), axis 'off';
colormap('gray')
title('Reconstructed face No. 3');

