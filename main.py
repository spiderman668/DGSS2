
import random
import os
import torch.cuda
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, \
    balanced_accuracy_score, f1_score
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn import metrics
from torch.utils.data import random_split, DataLoader
from Dataset import *
from model import *
import multiprocessing
#from torch.utils.tensorboard import SummaryWriter




SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(model, device, loader_train, optimizer, loss_fn, epoch):
    print('Training on {} samples...'.format(len(loader_train.dataset)))
    train_loss_avg = 0.
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, (molecule_left, molecule_right, cell_line, label) in enumerate(loader_train):
        molecule_left, molecule_right, cell_line, y = molecule_left.to(device), molecule_right.to(device), cell_line.to(device), label.long().to(device)
        optimizer.zero_grad()
        output = model(molecule_left, molecule_right, cell_line)
        loss = loss_fn(output, y)
        train_loss_avg = train_loss_avg + loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(molecule_left.to_data_list()),
                                                                           len(loader_train.dataset),
                                                                           100. * batch_idx / len(loader_train),
                                                                           loss.item()))
    return train_loss_avg / len(loader_train)


def predicting(model, device, loader_test, loader_train, epoch):
    val_loss = 0.
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()

    # total_preds_train = torch.Tensor()
    # total_labels_train = torch.Tensor()
    # total_prelabels_train = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader_test.dataset)))
    with torch.no_grad():
        for batch_idx, (molecule_left, molecule_right, cell_line, label) in enumerate(loader_test):
            molecule_left, molecule_right, cell_line, y = molecule_left.to(device), molecule_right.to(
                device), cell_line.to(device), label.long()
            output = model(molecule_left, molecule_right, cell_line).to('cpu')
            loss = loss_fn(output, y)
            val_loss = val_loss + loss.item()
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, y.view(-1, 1)), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten(), val_loss / len(loader_test)



if __name__ == '__main__':
    # writer = SummaryWriter('runs/without_all/leave_comb')
    # multiprocessing.set_start_method('spawn')
    modeling = CBA
    #writer = SummaryWriter('logs')
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 128
    LR = 0.0001
    LOG_INTERVAL = 200
    NUM_EPOCHS = 40

    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    comb_path = './data/hsa/leave_drug/'
    cell_path = os.path.join('./data', 'cell_lines.feather')
    print(modeling.__name__)

    auc_ls = []
    for fold in range(0, 5):
        print(f"Fold {fold + 1}:")
        print(f'use device: {device}')
        model = modeling().to(device)
        train_set_path = comb_path + f'train_{fold}.feather'
        test_set_path = comb_path + f'test_{fold}.feather'
        # 五折交叉
        train_set = DrugCombDataset(train_set_path, cell_path)
        test_set = DrugCombDataset(test_set_path, cell_path)
        loader_train = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True,pin_memory=True, num_workers=10,
                                  collate_fn=collate, drop_last=True)
        loader_test = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, shuffle=False,pin_memory=True, num_workers=10,
                                 collate_fn=collate, drop_last=True)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.1)

        file_result = './result2/hsa/leave_drug/fold_' + str(fold) + '.csv'
        file_result2 = './result2/hsa/leave_drug/fold_' + str(fold) + '_all' + '.csv'
        # model_path = f'./models/fold{fold}_best_model.pth'
        AUCs = ('Epoch,AUC_dev,PR_AUC,ACC,BACC,PREC,TPR,KAPPA,RECALL,Precision,F1')
        with open(file_result, 'w') as f:
            f.write(AUCs + '\n')
        with open(file_result2, 'w') as f:
            f.write(AUCs + '\n')

        best_auc = 0
        for epoch in range(NUM_EPOCHS):
            train_loss_avg = train(model, device, loader_train, optimizer, loss_fn, epoch + 1)
            #writer.add_scalar('Loss/Train', train_loss_avg, epoch + 1)
            print("第%d个epoch的学习率：%f" % (epoch + 1, optimizer.param_groups[0]['lr']))
            scheduler.step()
            T, S, Y, val_loss_avg = predicting(model, device, loader_test, loader_train, epoch + 1)
            #writer.add_scalar('Loss/Test', val_loss_avg, epoch + 1)
            # T is correct label
            # S is predict score
            # Y is predict label

            # compute preformence
            AUC = roc_auc_score(T, S)
            #writer.add_scalar('AUC', AUC, epoch + 1)
            precision, recall, threshold = metrics.precision_recall_curve(T, S)
            PR_AUC = metrics.auc(recall, precision)
            #writer.add_scalar('PR_AUC', PR_AUC, epoch + 1)
            BACC = balanced_accuracy_score(T, Y)
            #writer.add_scalar('BACC', BACC, epoch + 1)
            tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
            TPR = tp / (tp + fn)
            #writer.add_scalar('TPR', TPR, epoch + 1)
            PREC = precision_score(T, Y)
            #writer.add_scalar('PREC', PREC, epoch + 1)
            ACC = accuracy_score(T, Y)
            #writer.add_scalar('ACC', ACC, epoch + 1)
            KAPPA = cohen_kappa_score(T, Y)
            #writer.add_scalar('KAPPA', KAPPA, epoch + 1)
            recall = recall_score(T, Y)
            #writer.add_scalar('Recall', recall, epoch + 1)
            precision = precision_score(T, Y)
            #writer.add_scalar('Precision', precision, epoch + 1)
            F1 = f1_score(T, Y)
            #writer.add_scalar('F1', F1, epoch + 1)
            AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall, precision, F1]
            # save data
            if best_auc < AUC:
                best_auc = AUC
                save_AUCs(AUCs, file_result)
                # torch.save(model.state_dict(), model_path)


            print('test_auc:', AUC)
            print('best_auc:', best_auc)
            save_AUCs(AUCs, file_result2)
        auc_ls.append(best_auc)
        #writer.close()
    print(auc_ls)
