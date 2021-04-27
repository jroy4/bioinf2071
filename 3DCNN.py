#Basic brain age CNN, proof of concept (running on CPU)

import torch
import torch.nn as nn
import numpy as np
import math
from sklearn import metrics 
import os
from torch.autograd import Variable 
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import nibabel as nib
import pdb



columns = ['Site','ID','Gender','CHD','Age','Path']
data_dir='/Users/maxreynolds/Desktop/Bioinf2071/BrainAge/data'
images_dir=os.path.join(data_dir,'SVR_T1s_Prepped_70')
data_file=os.path.join(data_dir,'data.csv')



########## Models ###########
class CNN(nn.Module):
    #similar to https://github.com/Captain-Hong/Brain-Age-Prediction-of-Children/blob/master/3D%20CNN%20for%20predicting%20children%20brain%20age
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3),stride=(1, 1, 1), padding=(1, 1, 1),)
        self.bn1=nn.BatchNorm3d(32)
        self.active1=nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3))

        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3),stride=(1, 1, 1), padding=(1, 1, 1),)
        self.bn2=nn.BatchNorm3d(64)
        self.active2=nn.ReLU()
        self.conv22 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3),stride=(1, 1, 1), padding=(1, 1, 1),)
        self.bn22=nn.BatchNorm3d(64)
        self.active22=nn.ReLU()
        self.pool2=nn.MaxPool3d(kernel_size=(3, 3, 3))

        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3),stride=(1, 1, 1), padding=(1, 1, 1),)
        self.bn3=nn.BatchNorm3d(128)
        self.active3=nn.ReLU()
        self.pool3=nn.MaxPool3d(kernel_size=(3, 3,3))

        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3),stride=(1, 1, 1), padding=(1, 1, 1),)
        self.bn4=nn.BatchNorm3d(256)
        self.active4=nn.ReLU()
        self.pool4=nn.MaxPool3d(kernel_size=(3, 3, 3))
   
        self.fc1 = nn.Linear(256* 1 * 1, 128)
        self.active5=nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        self.active6=nn.ReLU()
        
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.active1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.active2(x)
        x = self.conv22(x)
        x = self.bn22(x)
        x = self.active22(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.active3(x)
        x = self.pool3(x)

        #added
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.active4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        x=self.fc1(x)
        x=self.active5(x)

        x=self.fc2(x)
        x=self.active6(x)

        output=self.fc3(x)
        return output

########## Helper calc functions (also from model github) ##########
# def computeCorrelation(X, Y):
#     xBar = np.mean(X)
#     yBar = np.mean(Y)
#     SSR = 0
#     varX = 0
#     varY = 0
#     for i in range(0, len(X)):
#         diffXXBar = X[i] - xBar
#         diffYYBar = Y[i] - yBar
#         SSR +=(diffXXBar * diffYYBar)
#         varX += diffXXBar**2
#         varY += diffYYBar**2
#     SST = math.sqrt(varX * varY)
#     return SSR/SST

# def rmse(y_test, y_pred):
#     return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# def mae(y_test, y_pred):
#     mae=metrics.mean_absolute_error(y_test, y_pred)
#     return mae

# def Get_Average(list):
#    sum = 0
#    for item in list:
#       sum += item
#    return sum/len(list) 


########## Datasets ###########
class BrainAgeDataset(Dataset):
    def __init__(self, csv_file, images_dir):
        """
        Args:
            csv_file: path to csv file with gender,age,site,etc...
            images_dir: directory with images

        """
        image_paths = [x for x in os.listdir(images_dir) if x.endswith(".nii.gz")]
        self.data = pd.read_csv(csv_file, header=None)
        self.data.columns=columns
        self.data['Path']=self.data['Path'].map(lambda x: x+'_001_prepped.nii.gz')

        #check all rows in csv have matching images
        assert(all(elem in os.listdir(images_dir) for elem in self.data['Path']))


        # pdb.set_trace()
        if not all(elem in list(self.data['Path']) for elem in os.listdir(images_dir)):
            num=sum(list(elem not in list(self.data['Path']) for elem in os.listdir(images_dir)))
            print('Warning:',num,'Image(s) in image_dir without matching rows in csv')
        

    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        data_row=self.data.loc[idx]
        img=nib.load(os.path.join(images_dir,data_row['Path']))
        img=torch.tensor(np.array(img.dataobj))
        return (img,np.array(data_row['CHD']),np.array(data_row['Age']))




device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_repeats=1





for repeat in range(num_repeats):
    print('*****setting hyperparameters*****')

    # EPOCH1 =40
    trainloss=[]
    # All_avg_trainloss=[]
    # All_avg_testloss=[]
    # RMSE_test=[]
    # RMSE_train=[]
    LR_record=[]
    LR=0.00001
    Decay=0
    # BATCH_SIZE =64
    # loss_func = nn.MSELoss()
    criterion = nn.MSELoss()
    iternum = 0
    num_epochs=50
    

    print('*****Loading data*****')
    all_brain_dataset=BrainAgeDataset(data_file,images_dir)
    train_data=all_brain_dataset
    test_data=all_brain_dataset
    trainloader=DataLoader(all_brain_dataset,batch_size=4,shuffle=True,num_workers=0)
    testloader=DataLoader(all_brain_dataset,batch_size=4,shuffle=True,num_workers=0) #Same as train for now
    
    pdb.set_trace()
    print('*****initializing network*****')
    cnn=CNN()
    cnn=cnn.to(device)

    print('*****training*****')
    for epoch in range(num_epochs):
        epoch_loss=0
        # LR=... Update LR based on decay (do this later)
        trainloss1=[]
        testloss1=[]
        trues=np.array([])
        preds=np.array([])
        
        optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=0.9)

        for i,data in enumerate(trainloader):
            
            optimizer.zero_grad()

            #data[1]is CHD, not using at the moment
            images,labels=torch.unsqueeze(data[0],1), data[2].unsqueeze(1)
            outputs=cnn(images)

            loss=criterion(outputs,labels.to(torch.float32))
            epoch_loss+=loss
            loss.backward()
            optimizer.step()
            preds=np.append(preds, outputs.detach().numpy())
            trues=np.append(trues,labels.detach().numpy().squeeze())
            # pdb.set_trace()

            # pdb.set_trace()
        print('eopch',epoch,'loss:',epoch_loss, 'mae:',metrics.mean_absolute_error(preds,trues))




