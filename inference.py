import random
import pandas as pd
import numpy as np
import os
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore') 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'VIDEO_LENGTH':50,
    'IMG_SIZE':128,
    'BATCH_SIZE':16,
    'SEED':41,
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

test = pd.read_csv('./test.csv')
submit = pd.read_csv('./sample_submission.csv')

class r3dDataSet(Dataset):
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


from decord import VideoReader
from einops import rearrange
from pytorchvideo.transforms.transforms_factory import create_video_transform
from transformers import AutoModel, AutoImageProcessor, AutoConfig

model_config = AutoConfig.from_pretrained(CFG['model_name'])
image_processor_config = AutoImageProcessor.from_pretrained(CFG['model_name'])


class timesDataSet(Dataset):
    def __init__(self, video_path_list, transform=None):
        self.video_path_list = video_path_list
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
        return self.frames_list[index]

    def __len__(self):
        return len(self.video_path_list)

val_transform = create_video_transform(
    mode='val',
    num_samples=model_config.num_frames,
    video_mean = tuple(image_processor_config.image_mean),
    video_std = tuple(image_processor_config.image_std),
    crop_size = tuple(image_processor_config.crop_size.values())
)

r3d_test_dataset = r3dDataSet(test['video_path'].values, None)
r3d_test_loader = DataLoader(r3d_test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

times_test_dataset = timesDataSet(test['video_path'].values, transform=val_transform)
times_test_loader = DataLoader(times_test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

class r3dModel(nn.Module):
    def __init__(self, num_classes):
        super(r3dModel, self).__init__()
        self.model1 = r3d_18(weights='KINETICS400_V1')
        self.fc = nn.Linear(400, num_classes)

    def forward(self, x):
        # from multi gpu to single gpu
        x = self.model1(x)
        x = self.fc(x)
        return x

class timesModel(nn.Module):
    def __init__(self, num_classes):
        super(timesModel, self).__init__()
        self.model1 = AutoModel.from_pretrained(CFG['model_name'])
        # self.model1.layernorm = nn.Identity()
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.model1(x).last_hidden_state.mean(dim=1)
        x = self.fc(x)
        return x

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            videos = videos.float().to(device)
            
            logit = model(videos)

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return preds

r3d_model = nn.DataParallel(r3dModel(num_classes=2), device_ids = list(range(2)), dim=0)
times_model = nn.DataParallel(timesModel(num_classes=3), device_ids = list(range(2)), dim=0)

model_path = '/home/hwlee/dacon/CAR/voting_model/'

crash_pred_list = []
ego_pred_list = []
timing_pred_list = []

for model_name in os.listdir(model_path):
    print(model_name)
    if 'crash' in model_name:
        r3d_model.load_state_dict(torch.load(model_path + model_name))
        crash_preds = inference(r3d_model, r3d_test_loader, device)
        crash_pred_list.append(crash_preds)
    elif 'ego-involve' in model_name:
        r3d_model.load_state_dict(torch.load(model_path + model_name))
        ego_preds = inference(r3d_model, r3d_test_loader, device)
        ego_pred_list.append(ego_preds)
    elif 'timing' in model_name:
        r3d_model.load_state_dict(torch.load(model_path + model_name))
        timing_preds = inference(r3d_model, r3d_test_loader, device)
        timing_pred_list.append(timing_preds)
    elif 'weather' in model_name:
        # times_model.load_state_dict(torch.load(model_path + model_name))
        # weather_preds = inference(times_model, times_test_loader, device)
        times_model.load_state_dict(torch.load(model_path + model_name))
        weather_preds = inference(times_model, r3d_test_loader, device)

crash_pred_list = np.array(crash_pred_list)
ego_pred_list = np.array(ego_pred_list)
timing_pred_list = np.array(timing_pred_list)

crash_pred_list = np.transpose(crash_pred_list)
ego_pred_list = np.transpose(ego_pred_list)
timing_pred_list = np.transpose(timing_pred_list)

crash_pred = np.array([np.argmax(np.bincount(crash_pred_list[i])) for i in range(len(crash_pred_list))])
ego_pred = np.array([np.argmax(np.bincount(ego_pred_list[i])) for i in range(len(ego_pred_list))])
timing_pred = np.array([np.argmax(np.bincount(timing_pred_list[i])) for i in range(len(timing_pred_list))])

submit['crash'] = crash_pred
submit['ego-involve'] = ego_pred
submit['timing'] = timing_pred
submit['weather'] = weather_preds

submit.loc[(submit['crash'] == 0), 'label'] = 0
submit.loc[(submit['crash'] == 1) & (submit['ego-involve'] == 1) & (submit['weather'] == 0) & (submit['timing'] == 0), 'label'] = 1
submit.loc[(submit['crash'] == 1) & (submit['ego-involve'] == 1) & (submit['weather'] == 0) & (submit['timing'] == 1), 'label'] = 2
submit.loc[(submit['crash'] == 1) & (submit['ego-involve'] == 1) & (submit['weather'] == 1) & (submit['timing'] == 0), 'label'] = 3
submit.loc[(submit['crash'] == 1) & (submit['ego-involve'] == 1) & (submit['weather'] == 1) & (submit['timing'] == 1), 'label'] = 4
submit.loc[(submit['crash'] == 1) & (submit['ego-involve'] == 1) & (submit['weather'] == 2) & (submit['timing'] == 0), 'label'] = 5
submit.loc[(submit['crash'] == 1) & (submit['ego-involve'] == 1) & (submit['weather'] == 2) & (submit['timing'] == 1), 'label'] = 6
submit.loc[(submit['crash'] == 1) & (submit['ego-involve'] == 0) & (submit['weather'] == 0) & (submit['timing'] == 0), 'label'] = 7
submit.loc[(submit['crash'] == 1) & (submit['ego-involve'] == 0) & (submit['weather'] == 0) & (submit['timing'] == 1), 'label'] = 8
submit.loc[(submit['crash'] == 1) & (submit['ego-involve'] == 0) & (submit['weather'] == 1) & (submit['timing'] == 0), 'label'] = 9
submit.loc[(submit['crash'] == 1) & (submit['ego-involve'] == 0) & (submit['weather'] == 1) & (submit['timing'] == 1), 'label'] = 10
submit.loc[(submit['crash'] == 1) & (submit['ego-involve'] == 0) & (submit['weather'] == 2) & (submit['timing'] == 0), 'label'] = 11
submit.loc[(submit['crash'] == 1) & (submit['ego-involve'] == 0) & (submit['weather'] == 2) & (submit['timing'] == 1), 'label'] = 12

submit.drop(['crash', 'ego-involve', 'weather', 'timing'], axis=1, inplace=True)

submit.to_csv('timing_voting_submit.csv', index=False)