import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


from option.config import *
from model.model_main import IQARegression
from model.backbone import resnet50_backbone
from trainer import train_epoch, eval_epoch
from utils.util import RandHorizontalFlip, Normalize, ToTensor, RandShuffle
from utils.gpu_util import GPUGet
from collections import OrderedDict

# config file
config = Config({'checkpoint': './weights/epoch50.pth'})

# data selection
if config.db_name == 'KonIQ-10k':
    from data.koniq import IQADataset

# dataset separation (8:2)
train_scene_list, test_scene_list = RandShuffle(config)
print('number of train scenes: %d' % len(train_scene_list))
print('number of test scenes: %d' % len(test_scene_list))

# data load
train_dataset = IQADataset(
    db_path=config.db_path,
    txt_file_name=config.txt_file_name,
    scale_1=config.scale_1,
    scale_2=config.scale_2,
    transform=transforms.Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), RandHorizontalFlip(), ToTensor()]),
    train_mode=True,
    scene_list=train_scene_list,
    train_size=config.train_size
)
test_dataset = IQADataset(
    db_path=config.db_path,
    txt_file_name=config.txt_file_name,
    scale_1=config.scale_1,
    scale_2=config.scale_2,
    transform= transforms.Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensor()]),
    train_mode=False,
    scene_list=test_scene_list,
    train_size=config.train_size
)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=True)

# 获取device_ids
# device_ids = [0, 1, 2, 3]
gpu_get = GPUGet()
device_ids = gpu_get.get_available_gpus()
print(device_ids)

# 基础模型创建
model_backbone = resnet50_backbone()
model_transformer = IQARegression(config)

# loss function & optimization
criterion = torch.nn.L1Loss()
params = list(model_backbone.parameters()) + list(model_transformer.parameters())
optimizer = torch.optim.SGD(params, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)
#optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

# load weights & optimizer
if config.checkpoint is not None:
    checkpoint = torch.load(config.checkpoint)
    
    # 加载Backbone（先加载到单卡模型，再包装为DataParallel）
    model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])  # 加载参数
    model_backbone.to(config.device)
    
    # 加载Transformer（同理）
    model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
    model_transformer.to(config.device)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
else:
    start_epoch = 0

if len(device_ids) >1:
    print("Use", len(device_ids), "GPUs")
    model_backbone = torch.nn.DataParallel(model_backbone, device_ids)
    model_transformer = torch.nn.DataParallel(model_transformer, device_ids)
    
# make directory for saving weights
if not os.path.exists(config.snap_path):
    os.mkdir(config.snap_path)

# train & validation
for epoch in range(start_epoch, config.n_epoch):
    loss, rho_s, rho_p = train_epoch(config, epoch, model_transformer, model_backbone, criterion, optimizer, scheduler, train_loader)

    if (epoch+1) % config.val_freq == 0:
        loss, rho_s, rho_p = eval_epoch(config, epoch, model_transformer, model_backbone, criterion, test_loader)