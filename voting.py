import random
import pandas as pd
import numpy as np
import os
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torchvision.models.video import r3d_18
import warnings
warnings.filterwarnings(action='ignore') 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'VIDEO_LENGTH':50,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':8,
    'SEED':2001,
    'IMG_SIZE':128,
    'EPOCHS':30
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])

df = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
submit = pd.read_csv('./sample_submission.csv')

df['crash'] = df['label'].apply(lambda x: int(0) if x == int(0) else 1)
df['ego-involve'] = df['label'].apply(lambda x: int(1) if x in [1,2,3,4,5,6] else int(0) if x in [7,8,9,10,11,12] else None)
df['weather'] = df['label'].apply(lambda x: int(0) if x in [1,2,7,8] else int(1) if x in [3,4,9,10] else int(2) if x in [5,6,11,12] else None)
df['timing'] = df['label'].apply(lambda x: int(0) if x in [1,3,5,7,9,11] else int(1) if x in [2,4,6,8,10,12] else None)

class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list):
        self.video_path_list = video_path_list
        self.label_list = label_list
        self.frames_list = []

        for video in tqdm(self.video_path_list):
            sub_frames = []
            cap = cv2.VideoCapture(video)
            for aa in range(CFG['VIDEO_LENGTH']):
                _, img = cap.read()
                img = cv2.resize(img, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
                img = img / 255.
                sub_frames.append(img)
            frame_torch = torch.FloatTensor(np.array(sub_frames)).permute(3, 0, 1, 2)
            self.frames_list.append(frame_torch)
        
    def __getitem__(self, index):
        frames = self.frames_list[index]
        
        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames
        
    def __len__(self):
        return len(self.video_path_list)



class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.model1 = r3d_18(weights='KINETICS400_V1')
        self.fc = nn.Linear(400, num_classes)

    def forward(self, x):
        x = self.model1(x)
        x = self.fc(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction=self.reduction)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1-pt)**self.gamma * ce_loss).mean()
        return focal_loss

def train(model, optimizer, train_loader, val_loader, scheduler, class_weights, device):
    model = nn.DataParallel(model, device_ids = list(range(2)), dim=0)
    model.to(device)
    criterion = FocalLoss(weight=class_weights).to(device)
    
    best_val_loss = 99999
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for videos, labels in tqdm(iter(train_loader)):
            videos = videos.float().to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            
            output = model(videos)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')
        
        if scheduler is not None:
            scheduler.step(_val_score)
            
        if best_val_score <= _val_score:
                best_val_loss = _val_loss
                best_val_score = _val_score
                best_model = model

    print(f'Best Val Loss : [{best_val_loss:.5f}] Best Val F1 : [{best_val_score:.5f}]')

    return best_model, best_val_loss, best_val_score

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, trues = [], []
    
    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            labels = labels.type(torch.LongTensor)
            videos = videos.to(device)
            labels = labels.to(device)
            
            logit = model(videos)
            
            loss = criterion(logit, labels)
            
            val_loss.append(loss.item())
            
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
        
        _val_loss = np.mean(val_loss)
    
    _val_score = f1_score(trues, preds, average='macro')
    # _val_score = accuracy_score(trues, preds)
    return _val_loss, _val_score

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            videos = videos.to(device)
            
            logit = model(videos)

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return preds

info = ['crash', 'ego-involve', 'timing']

test_dataset = CustomDataset(test['video_path'].values, None)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

for infos in info:
    if infos == 'crash':
        df1 = df
    else:
        df1 = df[df['label'] != 0].reset_index(drop=True)

    if infos == 'weather': SPLITS = 1
    else: SPLITS = 5

    print(infos, 'start', '='*50)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG['SEED'])
    for n, (train_index, val_index) in enumerate(skf.split(df1, df1['label'])):
        df1.loc[val_index, 'fold'] = int(n)
    df1['fold'] = df1['fold'].astype(int)
    
    pred_list = []
    models = []

    for fold in range(SPLITS):
        print(f'Fold {fold+1} start')

        if infos == 'weather':
            train1 = df1[df1['fold'] != 0].reset_index(drop=True)
            val = df1[df1['fold'] == 0].reset_index(drop=True)
        else:
            train1 = df1[df1['fold'] != fold].reset_index(drop=True)
            val = df1[df1['fold'] == fold].reset_index(drop=True)
            
        labels = []
        for i in df1[infos]:
            labels.append(int(i))
        labels.sort()
            
        if infos == 'crash': model = BaseModel(2)
        elif infos == 'ego-involve': model = BaseModel(2)
        elif infos == 'weather': model = BaseModel(3)
        elif infos == 'timing': model = BaseModel(2)
                
        class_weights = compute_class_weight(class_weight = "balanced", classes=np.unique(labels), y=labels)
        class_weights = torch.FloatTensor(class_weights).to(device)

        train_dataset = CustomDataset(train1['video_path'].values, train1[infos].values)
        train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

        val_dataset = CustomDataset(val['video_path'].values, val[infos].values)
        val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
            
        print("RAM Loading success")
            
        model.eval()

        optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)
        infer_model, best_loss, best_score = train(model, optimizer, train_loader, val_loader, scheduler, class_weights, device)

        train_dataset, train_loader = None, None
        val_dataset, val_loader = None, None

        torch.save(infer_model.module.state_dict(), f'/home/hwlee/dacon/CAR/voting_model/{infos}_model_{fold}.pt')
