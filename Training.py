import torch as tch
import torch.nn as nn
import wandb
import copy
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

bceLoss = nn.BCELoss()
device = tch.device("cuda:0" if tch.cuda.is_available() else "cpu")


def loss_bce(predictedVal, clinicalParam, lossParams):
    clinicalParam = (clinicalParam < lossParams['threshold']).float()
    return bceLoss(predictedVal, clinicalParam)

def evaluate(network, dataloader, lossFun, lossParams, args, getNoise=False):
    network.eval()
    with tch.no_grad():
        running_loss = 0.
        batchSize = dataloader.batch_size
        allParams = tch.empty(0).to(device)
        allPredictions=tch.empty(0).to(device)
        allNoiseVals = np.empty((0, 8))

        for ecg, clinicalParam in dataloader:
            ecg = ecg.to(device)
            clinicalParam = clinicalParam.to(device).unsqueeze(1)
            predictedVal = network(ecg)
            lossVal = lossFun(predictedVal, clinicalParam, lossParams)
            running_loss += lossVal.item()
            allParams = tch.cat((allParams, clinicalParam.squeeze()))
            allPredictions = tch.cat((allPredictions, predictedVal.squeeze()))

        running_loss = running_loss/len(dataloader)
    return running_loss, allParams, allPredictions, allNoiseVals

def trainNetwork(network, trainDataLoader, testDataLoader, numEpoch, optimizer, lossFun, lossParams, modelSaveDir, label, args, logToWandB=True, problemType='Binary'):
    print(f'Beginning Training for {label}')
    prevTrainingLoss = 0.0
    best_auc_test = 0.5

    for ep in range(numEpoch):
        print(f'Epoch {ep+1} of {numEpoch}')
        running_loss = 0.0
        network.train()
        count = 0
        for ecg, clinicalParam in trainDataLoader:
            print(f'Running through training batches {count} of {len(trainDataLoader)}', end='\r')

            count += 1
            optimizer.zero_grad()
            currBatchSize = ecg.shape[0]
            ## Input to GPU
            with tch.set_grad_enabled(True):
                ecg = ecg.to(device)
                clinicalParam = clinicalParam.to(device).unsqueeze(1)
                
                predictedVal = network(ecg)
                lossVal = lossFun(predictedVal, clinicalParam, lossParams)
                lossVal.backward()
                optimizer.step()
                running_loss = running_loss + lossVal

        currTrainingLoss = running_loss/len(trainDataLoader.dataset)

        print(f"Epoch {ep+1} train loss {currTrainingLoss}, Diff {currTrainingLoss - prevTrainingLoss}")
        prevTrainingLoss = currTrainingLoss
        print('Evalving Test')
        currTestLoss, allParams_test, allPredictions_test, _ = evaluate(network, testDataLoader, lossFun, lossParams, args)
        print('Evalving Train')
        currTrainLoss, allParams_train, allPredictions_train, _ = evaluate(network, trainDataLoader, lossFun, lossParams, args)
        print(f"Train Loss: {currTrainLoss} \n Test Loss: {currTestLoss}")
        allParams_train = (allParams_train.clone().detach().cpu() < lossParams['threshold']).long().numpy()
        allPredictions_train = allPredictions_train.clone().detach().cpu().numpy()
        allParams_test = (allParams_test.clone().detach().cpu() < lossParams['threshold']).long().numpy()
        allPredictions_test = allPredictions_test.clone().detach().cpu().numpy()

        if problemType == 'Binary':
            falsePos_train, truePos_train, _ = metrics.roc_curve(allParams_train, allPredictions_train)
            falsePos_test, truePos_test, _ = metrics.roc_curve(allParams_test, allPredictions_test)
            auc_train = metrics.roc_auc_score(allParams_train, allPredictions_train)
            auc_test = metrics.roc_auc_score(allParams_test, allPredictions_test)
        elif problemType == 'Regression':
            r2_train = metrics.r2_score(allParams_train, allPredictions_train)
            r2_test = metrics.r2_score(allParams_test, allPredictions_test)
        print(f'Train AUC: {auc_train:0.3f} test AUC: {auc_test:0.3f}')
        
        if auc_test > best_auc_test:
            print('Saving Model')
            best_auc_test = auc_test
            best_model = copy.deepcopy(network.state_dict())
            tch.save(best_model, modelSaveDir+label+'.pt')
            saved=1
        else:
            saved=0
        
        if logToWandB:
            print('Logging to wandb')
            plt.figure(1)
            fig, ax1 = plt.subplots(1, 2)
            if problemType == 'Binary':
                print(f'Train AUC: {auc_train:0.3f} test AUC: {auc_test:0.3f}')
                ax1[0].plot(falsePos_train, truePos_train)
                ax1[0].set_title(f'ROC train, AUC: {auc_train:0.3f}')
                ax1[1].plot(falsePos_test, truePos_test)
                ax1[1].set_title(f'ROC Test, AUC: {auc_test:0.3f}')
                plt.suptitle(f'ROC curves train AUC: {auc_train:0.3f} test AUC: {auc_test:0.3f}')

                print('Figures Made')
                wandb.log({
                    'Epoch':ep,
                    'Training Loss': currTrainingLoss,
                    'Test Loss': currTestLoss,
                    'Saved Best': saved,
                    'auc test': auc_test,
                    'auc train': auc_train,
                    'ROCs individual': plt
                })
        elif problemType == 'Regression':
            print(f'R2 Train: {r2_train: 0.3f} R2 test: {r2_test:0.3f}')
            allParams_train, sortIx = tch.sort(allParams_train)
            allPredictions_train = allPredictions_train[sortIx]
            allParams_test, sortIx = tch.sort(allParams_test)
            allPredictions_test = allPredictions_test[sortIx]
            ax1[0].plot(allParams_train,allPredictions_train)
            ax1[0].set_title(f'Regression train, R2: {r2_train:0.3f}')
            ax1[1].plot(allParams_test,allPredictions_test)
            ax1[1].set_title(f'Regression test, R2: {r2_test:0.3f}')
            plt.suptitle(f'Regression Curves R2 train: {r2_train:0.3f} R2 test: {r2_test:0.3f}')
            print('figures made')
            wandb.log({'Epoch':ep,
                        'Training Loss':currTrainLoss,
                        'Test Loss':currTestLoss,
                        'saved best':saved,
                        'r2 test':r2_test,
                        'r2 train':r2_train,
                        'Regression Plots':plt})

        if ep+1 >= numEpoch:
            print('Saving end Model')
            final_model = copy.deepcopy(network.state_dict())
            tch.save(final_model, modelSaveDir+label+'_final.pt')