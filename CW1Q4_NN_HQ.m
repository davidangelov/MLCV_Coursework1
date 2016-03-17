% MLCV coursework 1
% Q4 HQ

clc;
clear;
close all;

load face.mat
% 520 images of size 56x46

correctRate_record = [];
train_time_record = [];
test_time_record = [];

% for M = 1:100
M = 50;

    correctRate_sum = 0;
    train_time = 0;
    test_time = 0;
    
%     for iter = 1:50
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
    tic;

    % mean face
    averageFace = mean(Xtrain,2);

    % sustract mean face
    averageFace = repmat(averageFace, [1,416]);
    A = Xtrain - averageFace;

    % covariance matrix (1/N)AT*A
    S = A.'*A/416;

    % eigenvector of S
    [eigVector, eigValue] = eig(S);
    eigValue = diag(eigValue);
    [eigValueSort, sortID] = sort(eigValue,'descend');

    eigFaces = eigVector(:,sortID(1:M));
    eigFacesU = A*eigFaces;

    % normalization
    for idU = 1:M

        eigFacesU(:,idU) = eigFacesU(:,idU)./norm(eigFacesU(:,idU));

    end
    %% Representing faces onto eigenfaces

    % The columns of the matrix are projections
    wMatrix = (A.'*eigFacesU).';

    train_time = train_time + toc;

    %% Testing
    tic;

    phiTest = Xtest - averageFace(:,1:size(Xtest,2));

    wTest = (phiTest.'*eigFacesU).';

    [predictedTrainID, en] = fNN(wTest,wMatrix);
    % convert position to ID
    predictedID = ceil(predictedTrainID/8);

    trueID = reshape(repmat(1:52,2,1),1,52*2);

    x = find(predictedID == trueID);
    correctRate = length(find(predictedID == trueID))/104;
    correctRate_sum = correctRate_sum + correctRate;

    test_time = test_time + toc;
    
%     end
    
%     correctRate_record = [correctRate_record, correctRate_sum/iter];
%     train_time_record = [train_time_record, train_time/iter];
%     test_time_record = [test_time_record, test_time/iter];
    
% end

%% Failed case
xf1 = Xtest(:,4);
xmc1 = Xtrain(:,5*8+5);

xf2 = Xtest(:,21);
xmc2 = Xtrain(:,11*8+5);

figure
subplot(221)
imshow(uint8(reshape(xf1, 56, 46)), 'Border','tight');
colormap('gray'),xlabel('(a)');

subplot(222)
imshow(uint8(reshape(xmc1, 56, 46)), 'Border','tight');
colormap('gray'),xlabel('(b)');

subplot(223)
imshow(uint8(reshape(xf2, 56, 46)), 'Border','tight');
colormap('gray'),xlabel('(c)');

subplot(224)
imshow(uint8(reshape(xmc2, 56, 46)), 'Border','tight');
colormap('gray'),xlabel('(d)');

%% plotting results

% vM = 1:100;
% figure(1) % correction rate
% plot(vM, correctRate_record);
% xlabel('Number of M'), ylabel('Averaged correction rate');
% ylim([0 0.8]); grid on
% 
% figure(2) % train and test time
% plot(vM, train_time_record,'b', vM, test_time_record, 'r',...
%     vM, train_time_record + test_time_record, 'k');
% xlabel('Number of M'), ylabel('Averaged time');
% legend('Training time', 'Testing time', 'Total time');
% grid on

% %% confusion matrix
% figure(3)
% fig3 = figure(3);
% conf_mat = confusionmat(trueID, predictedID);
% imagesc(conf_mat); colorbar;
% xlabel('Target class'), ylabel('Predicted class');

