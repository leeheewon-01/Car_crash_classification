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
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':8,
    'SEED':2001,
    'IMG_SIZE':128,
    'EPOCHS':30,
    'model_name':'facebook/timesformer-base-finetuned-k400'
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

df['weather'] = df['label'].apply(lambda x: int(0) if x in [1,2,7,8] else int(1) if x in [3,4,9,10] else int(2) if x in [5,6,11,12] else None)

from decord import VideoReader
from einops import rearrange
from pytorchvideo.transforms.transforms_factory import create_video_transform
from transformers import AutoModel, AutoImageProcessor, AutoConfig

model_config = AutoConfig.from_pretrained(CFG['model_name'])
image_processor_config = AutoImageProcessor.from_pretrained(CFG['model_name'])

class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list, transform=None):
        self.video_path_list = video_path_list
        self.label_list = label_list
        self.transform = transform
        self.frames_list = []

        for vidio in tqdm(self.video_path_list):
            vr = VideoReader(vidio, width=CFG['IMG_SIZE'], height=CFG['IMG_SIZE'])
            video = torch.from_numpy(vr.get_batch(range(50)).asnumpy())
            video = rearrange(video, 't h w c -> c t h w')
            if self.transform:
                video = self.transform(video)
            video = rearrange(video, 'c t h w -> t c h w')
            self.frames_list.append(video)

    def __getitem__(self, index):
        video = self.frames_list[index]
        label = self.label_list[index]
        
        return video, label

    def __len__(self):
        return len(self.video_path_list)


"""class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list, transform=None):
        self.video_path_list = video_path_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, index):
        vr = VideoReader(self.video_path_list[index], width=CFG['IMG_SIZE'], height=CFG['IMG_SIZE'])
        video = torch.from_numpy(vr.get_batch(range(50)).asnumpy())
        video = rearrange(video, 't h w c -> c t h w')
        if self.transform:
            video = self.transform(video)
        video = rearrange(video, 'c t h w -> t c h w')
        label = self.label_list[index]
        
        if self.label_list is None:
            return video
        else:
            return video, label

    def __len__(self):
        return len(self.video_path_list)
"""
train_transform = create_video_transform(
    mode='train',
    num_samples=model_config.num_frames,
    video_mean = tuple(image_processor_config.image_mean),
    video_std = tuple(image_processor_config.image_std),
    crop_size = tuple(image_processor_config.crop_size.values())
)

val_transform = create_video_transform(
    mode='val',
    num_samples=model_config.num_frames,
    video_mean = tuple(image_processor_config.image_mean),
    video_std = tuple(image_processor_config.image_std),
    crop_size = tuple(image_processor_config.crop_size.values())
)


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.model1 = AutoModel.from_pretrained(CFG['model_name'])
        # using indentity to get the last hidden state
        self.model1.pooler = nn.Identity()
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.model1(x).last_hidden_state.mean(dim=1)
        x = self.fc(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction=self.reduction, weight=self.weight)
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

    return best_model, best_val_score

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

scores_list = []
model_list = []

df1 = df[df['label'] != 0].reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG['SEED'])
for n, (train_index, val_index) in enumerate(skf.split(df1, df1['label'])):
    df1.loc[val_index, 'fold'] = int(n)
df1['fold'] = df1['fold'].astype(int)

for fold in range(5):

    print(f'Fold : {fold}')
    
    train1 = df1[df1['fold'] != fold].reset_index(drop=True)
    val = df1[df1['fold'] == fold].reset_index(drop=True)
        
    labels = []
    for i in df1['weather']:
        labels.append(int(i))
    labels.sort()

    model = BaseModel(3)
            
    class_weights = compute_class_weight(class_weight = "balanced", classes=np.unique(labels), y=labels)
    class_weights = torch.FloatTensor(class_weights).to(device)

    train_dataset = CustomDataset(train1['video_path'].values, train1['weather'].values, transform = train_transform)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    val_dataset = CustomDataset(val['video_path'].values, val['weather'].values, transform = val_transform)
    val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
        
    print("RAM Loading success")
        
    model.eval()

    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)
    
    infer_model, val_score = train(model, optimizer, train_loader, val_loader, scheduler, class_weights, device)

    train_dataset, train_loader = None, None
    val_dataset, val_loader = None, None

    scores_list.append(val_score)
    model_list.append(infer_model)
# score_list 중 가장 높은 값의 index에 해당하는 모델을 best_model로 저장
best_model = model_list[scores_list.index(max(scores_list))]
torch.save(best_model.state_dict(), f'/home/hwlee/dacon/CAR/voting_model/weather_model_k400.pt')