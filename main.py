import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import ChestXrayDataSet
from model import DenseNet121
from sklearn.metrics import roc_auc_score

small_set = False

num_classes = 14
class_names = labels = ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema','Fibrosis','Effusion',
                        'Pneumonia','Pleural_Thickening','Cardiomegaly','Mass','Nodule','Hernia']
data_dir = 'data_entry_labels.csv'
batch_size = 32
max_epoch = 50

print('Loading Model')
model = DenseNet121(num_classes).cuda()

normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

ts = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

if small_set == True:
    train_list = 'train_list_small.txt'
    val_list = 'val_list_small.txt'
else:
    train_list = 'train_list.txt'
    val_list = 'val_list.txt'

print("Loading DataSet + DataLoader")
train_ds =  ChestXrayDataSet(data_dir=data_dir,
                             image_list_file=train_list,
                             transform=ts
                            )

train_loader = DataLoader(dataset=train_ds, batch_size=16,
                             shuffle=False)

val_ds =  ChestXrayDataSet(data_dir=data_dir,
                             image_list_file=val_list,
                             transform=ts
                            )

val_loader = DataLoader(dataset=val_ds, batch_size=8,
                             shuffle=False)

optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')

loss_fn = torch.nn.BCELoss(reduction='mean')

checkpoint = None

print('Starting Training Loop')
if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])


# TODO: WRITE TRAINING LOOP

lossMIN = 100000

val_loss = []

for epochID in range (0, max_epoch):
    tsTime = time.strftime("%H%M%S")
    tsDate = time.strftime("%d%m%Y")
    tsStart = tsDate + '-' + tsTime
    
    model.train()
    for batchID, (X, Y) in enumerate(train_loader):
        Y = Y.cuda()
        X = X.cuda()
        output = model(X)
        loss = loss_fn(output, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
    lossVal = 0
    lossValNorm = 0
    lossTensorMean = 0
    
    with torch.no_grad():
        for batchID, (X, Y) in enumerate(val_loader):
            
            Y = Y.cuda()
            X = X.cuda()
            output = model(X)
            loss = loss_fn(output, Y)
            lossTensorMean += loss
            lossVal += loss.item()
            lossValNorm += 1

        outLoss = lossVal / lossValNorm
        lossTensorMean = lossTensorMean / lossValNorm
    
    scheduler.step(lossTensorMean.item())
    
    tsTime = time.strftime("%H%M%S")
    tsDate = time.strftime("%d%m%Y")
    timestampEND = tsDate + '-' + tsTime
    
    val_loss.append(outLoss)

    # if outLoss < lossMIN:
    #     lossMIN = outLoss
    #     torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + tsStart + '.pth.tar')
    #     print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
    # else:
    #     print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))


print("Done Training Model")        
torch.save(model.state_dict(), 'model_dict.pt')

# # TODO: REMOVE LAST LAYER 

# # TODO: OUTPUT FEATURES VECTORS

# # TODO: GRAPH FEATURE VECTORS USING PCA/UMAP, K-MEANS CLUSTERING TO IDENTIFY SUBCLASS INFORMATION