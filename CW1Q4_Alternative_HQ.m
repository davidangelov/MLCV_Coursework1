% MLCV coursework 1
% Q4 DA

clc;
clear;
close all;

load face.mat
% 520 images of size 56x46

correctRate_record = [];
train_time_record = [];
test_time_record = [];

for M = 1:8
    
    correctRate_sum = 0;
    train_time = 0;
    test_time = 0;
    
    for iter = 1:50
        
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

%     M = 8;
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

    train_time = train_time + toc;
    %% Testing
    
    tic;
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

    Error_Array = pdist2(Xtest.',XReconstruct.');
    [~,Min_error_Test_ID] = min(Error_Array);

    predictedID = ceil(Min_error_Test_ID/2);

    trueID = reshape(repmat(1:52,2,1),1,52*2);

    x = find(predictedID == trueID);
    correctRate = length(find(predictedID == trueID))/104;
    correctRate_sum = correctRate_sum + correctRate;
    
    test_time = test_time + toc;
    
    end
    
    correctRate_record = [correctRate_record, correctRate_sum/iter];
    train_time_record = [train_time_record, train_time/iter];
    test_time_record = [test_time_record, test_time/iter];
    
end


% display(correctRate, 'Rate of correct prediction');

%% plotting results

vM = 1:8;
figure(1) % correction rate
plot(vM, correctRate_record);
xlabel('Number of M'), ylabel('Averaged correction rate');
ylim([0 1]); grid on

figure(2) % train and test time
plot(vM, train_time_record,'b', vM, test_time_record, 'r', ...
    vM, train_time_record + test_time_record, 'k');
xlabel('Number of M'), ylabel('Averaged time');
legend('Training time', 'Testing time', 'Total time');
grid on

%% confusion matrix
figure(3)
fig3 = figure(3);
conf_mat = confusionmat(trueID, predictedID);
imagesc(conf_mat); colorbar;
xlabel('Target class'), ylabel('Predicted class');

% % reconstructed testing image
% figure('Name','The first 20 reconstructed testing faces')
% for iX = 1:20
%     subplot(4,5,iX)
%     Xdisplay = reshape(XReconstruct(:,iX*2),[56,46]);
%     imagesc(Xdisplay),colormap('gray'); 
%     axis 'off'
% end
% 
% % original testing image
% figure('Name','The first 20 testing faces')
% for iXtest = 1:20
%     subplot(4,5,iXtest);
%     Xdisplay = reshape(Xtest(:,iXtest*2),[56,46]);
%     imagesc(Xdisplay),colormap('gray');
%     axis 'off'
% end
