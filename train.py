import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.utils.data import DataLoader


from option.config import Config
from model.model_main import IQARegression
from model.backbone import resnet50_backbone
from trainer import train_epoch, eval_epoch
from utils.util import RandHorizontalFlip, Normalize, ToTensor, RandShuffle


# 初始化分布式参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', -1))
args = parser.parse_args()
local_rank = args.local_rank
print(f"using cuda:%d" % local_rank)

# config file
config = Config({
    # device
    'gpu_id': args.local_rank,                          # specify GPU number to use
    'num_workers': 12,

    # data
    'db_name': 'KonIQ-10k',                                     # database type
    'db_path': './dataset/koniq-10k',                           # root path of database
    'txt_file_name': './IQA_list/koniq-10k.txt',                # list of images in the database
    'train_size': 0.8,                                          # train/vaildation separation ratio
    'scenes': 'all',                                            # using all scenes
    'scale_1': 384,                                             
    'scale_2': 224,
    'batch_size': 8,
    'patch_size': 32,

    # ViT structure
    'n_enc_seq': 32*24 + 12*9 + 7*5,        # input feature map dimension (N = H*W) from backbone
    'n_layer': 14,                          # number of encoder layers
    'd_hidn': 384,                          # input channel of encoder (input: C x N)
    'i_pad': 0,
    'd_ff': 384,                            # feed forward hidden layer dimension
    'd_MLP_head': 1152,                     # hidden layer of final MLP
    'n_head': 6,                            # number of head (in multi-head attention)
    'd_head': 384,                          # channel of each head -> same as d_hidn
    'dropout': 0.1,                         # dropout ratio
    'emb_dropout': 0.1,                     # dropout ratio of input embedding
    'layer_norm_epsilon': 1e-12,
    'n_output': 1,                          # dimension of output
    'Grid': 10,                             # grid of 2D spatial embedding

    # optimization & training parameters
    'n_epoch': 100,                         # total training epochs
    'learning_rate': 1e-4,                  # initial learning rate
    'weight_decay': 0,                      # L2 regularization weight
    'momentum': 0.9,                        # SGD momentum
    'T_max': 3e4,                           # period (iteration) of cosine learning rate decay
    'eta_min': 0,                           # minimum learning rate
    'save_freq': 10,                        # save checkpoint frequency (epoch)
    'val_freq': 5,                          # validation frequency (epoch)


    # load & save checkpoint
    'snap_path': './weights',               # directory for saving checkpoint
    'checkpoint': './weights/epoch10.pth',                     # load checkpoint
})


if local_rank != -1:
    # 初始化进程组
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    world_size = dist.get_world_size()
    print(world_size)
    rank = dist.get_rank()
    # 设备设置必须放在初始化进程组之后
    config.device = torch.device(f'cuda:{local_rank}')

# 动态设置batch_size
config.batch_size = 8 * world_size

# data selection
if config.db_name == 'KonIQ-10k':
    from data.koniq import IQADataset

# dataset separation (8:2)
train_scene_list, test_scene_list = RandShuffle(config)

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

# 数据集部分修改
train_sampler = DistributedSampler(train_dataset, shuffle=True)
test_sampler = DistributedSampler(test_dataset, shuffle=False)

# 调整DataLoader：注意shuffle=False，sampler替换为分布式采样器
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size // world_size,
    sampler=train_sampler,
    num_workers=config.num_workers,
    drop_last=True,
    shuffle=False,
    pin_memory=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=config.batch_size // world_size,  
    sampler=test_sampler,
    num_workers=config.num_workers,
    drop_last=False,  # 评估时允许保留不完整批次
    shuffle=False,
    pin_memory=True
)

# 合并模型以适配DDP
class CompleteModel(torch.nn.Module):
    def __init__(self, backbone, transformer):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
    
    def forward(self, mask_inputs, d_img_org, d_img_scale_1, d_img_scale_2):
        # 前向传播逻辑合并
        feat_org = self.backbone(d_img_org)
        feat_scale1 = self.backbone(d_img_scale_1)
        feat_scale2 = self.backbone(d_img_scale_2)
        return self.transformer(mask_inputs, feat_org, feat_scale1, feat_scale2)

# 实例化并包装模型
model_backbone = resnet50_backbone().to(config.device)
model_transformer = IQARegression(config).to(config.device)
complete_model = CompleteModel(model_backbone, model_transformer).to(config.device)
ddp_model = DDP(complete_model, device_ids=[local_rank])

# 优化器和损失函数
criterion = torch.nn.L1Loss()
params = list(ddp_model.parameters())
optimizer = torch.optim.SGD(params, lr=config.learning_rate * world_size, weight_decay=config.weight_decay, momentum=config.momentum)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

# 加载检查点
if config.checkpoint is not None and os.path.isfile(config.checkpoint):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}  # 多GPU加载映射
    checkpoint = torch.load(config.checkpoint, map_location=map_location)
    
    # 适配旧版本checkpoint参数名称
    if 'model_state_dict' in checkpoint: 
        ddp_model.module.load_state_dict(checkpoint['model_state_dict'])
    else:  # 兼容旧版模型的加载方式
        ddp_model.module.backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
        ddp_model.module.transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
else:
    start_epoch = 0

# 主进程创建保存目录
if rank == 0 and not os.path.exists(config.snap_path):
    os.makedirs(config.snap_path, exist_ok=True)

# train & validation
for epoch in range(start_epoch, config.n_epoch):
    # 每个epoch前设置sampler的epoch（保证shuffle正确性）
    train_loader.sampler.set_epoch(epoch)
    loss, rho_s, rho_p = train_epoch(config, epoch, ddp_model, criterion, optimizer, scheduler, train_loader)

    if (epoch+1) % config.val_freq == 0:
        val_loss, val_rho_s, val_rho_p = eval_epoch(config, epoch, ddp_model, criterion, test_loader)
    
    # 只由主进程保存模型
    if (epoch+1) % config.save_freq == 0 and rank == 0:
        save_path = os.path.join(
            config.snap_path, 
            f'epoch{epoch+1}_SROCC_{rho_s:.4f}_PLCC_{rho_p:.4f}.pth'
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': ddp_model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'SROCC': rho_s,
            'PLCC': rho_p
        }, save_path)
        print(f'Saved checkpoint to {save_path}')
