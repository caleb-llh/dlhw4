import os
import torch
import torchvision
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim

from sklearn.metrics import accuracy_score

from models import Net
import data
import parser

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)    

def train(mode, epoch, model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    preds = []
    gts = []
    for idx, (imgs, gt) in enumerate(train_loader):
        train_info = '[{}]Epoch: [{}][{}/{}]'.format(mode,epoch, idx+1, len(train_loader))
        
        if torch.cuda.is_available():
            imgs, gt = imgs.cuda(), gt.cuda()
        
        ''' forward path '''
        pred = model(imgs) # pred is now one-hot but gt is an integer

        ''' compute loss, backpropagation, update parameters '''
        loss = criterion(pred, gt)    # CE loss takes input (N,C) and targets (N)
        optimizer.zero_grad()         # set grad of all parameters to zero
        loss.backward()               # compute gradient for each parameters
        optimizer.step()              # update parameters
        
        total_loss += loss.data.cpu().numpy()
        train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())
        print("\r{}".format(train_info), end='')
    
    return total_loss/len(train_loader)


def test(epoch, model, test_loader, criterion):
    model.eval()
    total_loss = 0
    preds = []
    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(test_loader):
            if torch.cuda.is_available():
                imgs, gt = imgs.cuda(), gt.cuda()

            pred = model(imgs)

            ''' compute loss, backpropagation, update parameters '''
            loss = criterion(pred, gt) # compute loss
            total_loss += loss.data.cpu().numpy()

            ''' save preds and gts for accuracy calculation '''
            _, pred = torch.max(pred, dim = 1) #argmax to convert one-hot to integer
            # print(pred.shape, gt.shape)
            
            if torch.cuda.is_available():
                pred, gt = pred.cpu(), gt.cpu()
            pred = pred.numpy().squeeze()
            gt = gt.numpy().squeeze()
            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)
    return total_loss/len(test_loader), accuracy_score(gts, preds)

def main():
    ''' setup '''
    torch.manual_seed(args.random_seed)
    
    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_set = data.DATA(args,mode='train')
    val_set = data.DATA(args,mode='val')
    test_set = data.DATA(args,mode='test')
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                               batch_size=args.test_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)                             
    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=args.test_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)

    ''' load models '''
    print('===> prepare models ...')
    A = Net('A')
    B = Net('B')
    C = Net('C')
    models = {'A':A, 'B':B, 'C':C}

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    best_acc = {'A':0,'B':0,'C':0}
    best_epoch = {'A':0,'B':0,'C':0}
    train_losses = {'A':[],'B':[],'C':[]}
    val_losses = {'A':[],'B':[],'C':[]}
    val_accs = {'A':[],'B':[],'C':[]}
    test_losses = {'A':0,'B':0,'C':0}
    test_accs = {'A':0,'B':0,'C':0}
    
    ''' training and validation iterations'''
    for mode in ['A','B','C']:
        model = models[mode]
        ''' setup optimizer '''
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        if torch.cuda.is_available():
            model.cuda()
        
        for epoch in range(args.epoch):
            ''' train model and get averaged epoch loss '''
            train_loss = train(mode,epoch, model, train_loader, criterion, optimizer)       
            ''' evaluate the model '''
            val_loss, val_acc = test(epoch, model, val_loader, criterion)  
            train_losses[mode].append(train_loss)
            val_losses[mode].append(val_loss)
            val_accs[mode].append(val_acc)
            print('\nMode: {} Epoch: [{}] TRAIN_LOSS: {} VAL_LOSS: {} VAL_ACC:{}'.format(mode, epoch, train_loss, val_loss, val_acc))
            
            ''' save best model '''
            if val_acc > best_acc[mode]:
                save_model(model, os.path.join(args.save_dir, 'model_best_{}.pth.tar'.format(mode)))
                best_acc[mode] = val_acc
                best_epoch[mode] = epoch
        print("Mode: {} Best acc: {}, epoch {}".format(mode, best_acc, best_epoch[mode]))   

    ''' testing '''
    for mode in ['A','B','C']:
        model = models[mode]
        test_loss, test_acc = test(0, model, test_loader, criterion)
        test_losses[mode] = test_loss
        test_accs[mode] = test_acc
        print('Mode: {} TEST_LOSS:{} TEST_ACC:{}'.format(mode, test_loss, test_acc))

    ''' save train/val/test information as pickle files'''
    with open(os.path.join(args.save_dir,'train_losses.pkl'), 'wb') as f:
        pickle.dump(train_losses, f)
    with open(os.path.join(args.save_dir,'val_losses.pkl'), 'wb') as f:
        pickle.dump(val_losses, f)
    with open(os.path.join(args.save_dir,'val_accs.pkl'), 'wb') as f:
        pickle.dump(val_accs, f)
    with open(os.path.join(args.save_dir,'test_losses.pkl'), 'wb') as f:
        pickle.dump(test_losses, f)
    with open(os.path.join(args.save_dir,'test_accs.pkl'), 'wb') as f:
        pickle.dump(test_accs, f)

def plot():
    with open(os.path.join(args.save_dir,'train_losses.pkl'), 'rb') as f:
        train_losses = pickle.load(f)
    with open(os.path.join(args.save_dir,'val_losses.pkl'), 'rb') as f:
        val_losses = pickle.load(f)
    with open(os.path.join(args.save_dir,'val_accs.pkl'), 'rb') as f:
        val_accs = pickle.load(f)

    epochs = list(range(len(train_losses['A'])))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_losses['A'], 'o-', label="A", color='red')
    plt.plot(epochs, train_losses['B'], 'o-', label="B", color='blue')
    plt.plot(epochs, train_losses['C'], 'o-', label="C", color='green')
    plt.ylabel('Training Loss')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs, val_losses['A'], 'o-', label="A", color='red')
    plt.plot(epochs, val_losses['B'], 'o-', label="B", color='blue')
    plt.plot(epochs, val_losses['C'], 'o-', label="C", color='green')
    # plt.title('Comparing train and val accuracies')
    # plt.xlabel('Epochs')
    plt.ylabel('Validation Losses')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(epochs, val_accs['A'], 'o-', label="A", color='red')
    plt.plot(epochs, val_accs['B'], 'o-', label="B", color='blue')
    plt.plot(epochs, val_accs['C'], 'o-', label="C", color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracies')
    plt.legend()
    
    plt.savefig(os.path.join(args.save_dir,"train_graphs"), bbox_inches='tight')

if __name__=='__main__':
    args = parser.arg_parse()
    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # main()
    plot()
