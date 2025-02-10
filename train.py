import os
import torch
import logging
from torchvision import transforms
from torch.utils.data import DataLoader


from option.config import *
from model.model_main import IQARegression
from model.backbone import resnet50_backbone
from trainer import train_epoch, eval_epoch
from utils.util import RandHorizontalFlip, Normalize, ToTensor, RandShuffle
from utils.gpu_util import GPUGet
from utils.logger import TrainingLogger
from collections import OrderedDict

# config file
config = Config({
    'exp_name': 'koniq2',
    # 'checkpoint_name': 'epoch40.pth',
    })

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
    print("checkpoint %s 加载成功" % config.checkpoint)
    
    # 加载Backbone，包装为DataParallel
    model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])  # 加载参数
    
    # 加载Transformer，包装为DataParallel
    model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
else:
    start_epoch = 0
    print("checkpoint 未加载")

if len(config.device_ids) > 1:
    print("Use", len(config.device_ids), "GPUs")
    model_backbone = torch.nn.DataParallel(model_backbone, config.device_ids)
    model_transformer = torch.nn.DataParallel(model_transformer, config.device_ids)

# 模型创建完，移动到device上
model_backbone.to(config.device)
model_transformer.to(config.device)
    
# make directory for saving weights
exp_dir = os.path.join(config.snap_path, config.exp_name)
os.makedirs(exp_dir, exist_ok=True)                         # 创建实验目录
weights_dir = os.path.join(exp_dir, 'weights')  
os.makedirs(weights_dir, exist_ok=True)                     # 权重子目录
logs_dir = os.path.join(exp_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# create logger
if config.save_log:
    logger = TrainingLogger(os.path.join(logs_dir))
    train_log = logger.create_log(
        'train_log.csv', 
        ['epoch', 'loss', 'srocc', 'plcc', 'lr']
    )
    val_log = logger.create_log(
        'val_log.csv',
        ['epoch', 'loss', 'srocc', 'plcc', 'num_samples']
    )
    print('log文件创建成功！')

# train & validation
for epoch in range(start_epoch, config.n_epoch):
    train_metrics  = train_epoch(config, epoch, model_transformer, model_backbone, criterion, optimizer, scheduler, train_loader)

    # 始终记录训练日志
    TrainingLogger.add_record(train_log, train_metrics)
    
    if (epoch+1) % config.val_freq == 0:
        val_metrics = eval_epoch(config, epoch, model_transformer, model_backbone, criterion, test_loader)
        TrainingLogger.add_record(val_log, val_metrics)