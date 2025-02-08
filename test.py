import os
import time
import sys
import torch
from option.config import Config
from model.model_main import IQARegression
from model.backbone import resnet50_backbone
from trainer import train_epoch, eval_epoch
from utils.util import RandHorizontalFlip, Normalize, ToTensor, RandShuffle

class GPUGet:
    def __init__(self, min_gpu_number=1, time_interval=1, required_mem=20000, max_power=50):
        self.min_gpu_number = min_gpu_number
        self.time_interval = time_interval
        self.required_mem = required_mem  # 单位 MiB
        self.max_power = max_power        # 单位 W

    def get_gpu_info(self):
        """解析 nvidia-smi 输出，返回 GPU 状态字典"""
        gpu_status = os.popen('nvidia-smi | grep %').read().split('|')[1:]
        gpu_dict = {}
        for i in range(len(gpu_status) // 4):
            index = i * 4
            # 解析 GPU 状态、功率、显存
            status = str(gpu_status[index].split('   ')[2].strip())
            power = int(gpu_status[index].split('   ')[-1].split('/')[0].split('W')[0].strip())
            memory_used = int(gpu_status[index + 1].split('/')[0].split('M')[0].strip())
            memory_total = int(gpu_status[index + 1].split('/')[1].split('M')[0].strip())
            gpu_dict[i] = {
                "status": status,
                "power": power,
                "memory_used": memory_used,
                "memory_total": memory_total,
                "memory_available": memory_total - memory_used
            }
        return gpu_dict

    def find_available_gpus(self):
        """动态检测可用 GPU"""
        available_gpus = []
        while len(available_gpus) < self.min_gpu_number:
            gpu_dict = self.get_gpu_info()
            current_available = [
                gpu_id for gpu_id, info in gpu_dict.items()
                if info["memory_available"] >= self.required_mem and info["power"] <= self.max_power
            ]
            if len(current_available) >= self.min_gpu_number:
                available_gpus = current_available
                break
            else:
                print(f"等待可用 GPU... 当前可用: {current_available}")
                time.sleep(self.time_interval)
        return available_gpus

def test_model(config):
    # 加载模型
    model_backbone = resnet50_backbone().to(config.device)
    model_transformer = IQARegression(config).to(config.device)
    
    # 加载训练好的权重
    checkpoint = torch.load(config.checkpoint)
    model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
    model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
    
    # 评估模型
    loss, srcc, plcc = eval_epoch(config, 0, model_transformer, model_backbone, criterion, test_loader)
    print(f"Final Test Results - Loss: {loss:.4f}, SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")
    
if __name__ == "__main__":
    gpu_get = GPUGet()
    available_gpus = gpu_get.find_available_gpus()
    gpu_ids = ",".join(map(str, available_gpus))
    print(gpu_ids)
    
    config = Config({
        # device
        'gpu_id': "0",                          # specify GPU number to use
        'num_workers': 8,

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
        'checkpoint': './weights/epoch100.pth',                     # load checkpoint
    })
    # test_model(config)
    