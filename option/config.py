import json
import torch
from utils.gpu_util import GPUGet


# option/config.py
class Config(object):
    def __init__(self, custom_params=None):
        # 基础参数（可以通过custom_params修改）
        base_params = {
        # 硬件配置
        'gpu_get_class_path': 'utils.gpu_util.GPUGet',  # GPU检测类路径
        'gpu_id': None,                                 # 指定使用的GPU编号（None为自动选择）
        'num_workers': 4,                               # 数据加载线程数
        
        # 数据配置
        'db_name': 'KonIQ-10k',                         # 使用的数据库名称
        'db_path': './dataset/koniq-10k',              # 数据库根路径
        'txt_file_name': './IQA_list/koniq-10k.txt',   # 图像列表文件路径
        'train_size': 0.8,                              # 训练集划分比例
        'scenes': 'all',                                # 使用所有场景类型
        'scale_1': 384,                                 # 图像缩放尺寸1
        'scale_2': 224,                                 # 图像缩放尺寸2
        'batch_size': 32,                               # 批处理大小
        'patch_size': 32,                               # 图像块尺寸
        
        # ViT架构参数
        'n_enc_seq': 32*24 + 12*9 + 7*5,                # 输入特征图维度（N=H*W）
        'n_layer': 14,                                  # 编码器层数
        'd_hidn': 384,                                  # 编码器输入通道数
        'i_pad': 0,                                     # 填充索引
        'd_ff': 384,                                    # 前馈网络隐藏层维度
        'd_MLP_head': 1152,                             # 最终MLP隐藏层维度
        'n_head': 6,                                    # 多头注意力头数
        'd_head': 384,                                  # 每个注意力头的通道数
        'dropout': 0.1,                                 # 常规dropout比例
        'emb_dropout': 0.1,                             # 输入嵌入dropout比例
        'layer_norm_epsilon': 1e-12,                    # 层归一化epsilon值
        'n_output': 1,                                  # 输出维度
        'Grid': 10,                                     # 2D空间嵌入网格数
        
        # 训练参数
        'n_epoch': 100,                                 # 总训练轮数
        'learning_rate': 1e-4,                          # 初始学习率
        'weight_decay': 0,                              # L2正则化权重
        'momentum': 0.9,                                # SGD动量
        'T_max': 3e4,                                   # 余弦学习率衰减周期（迭代次数）
        'eta_min': 0,                                   # 最小学习率
        'save_freq': 10,                                # 保存检查点频率（按轮数）
        'val_freq': 5,                                  # 验证频率（按轮数）
        
        # 路径配置
        'snap_path': './weights',                      # 检查点保存目录
        'checkpoint': './weights/epoch40.pth',          # 预训练检查点路径
        
        # 日志记录
        'experiment_name': 'koniq_base',  # 实验唯一标识
        'log_dir': './training_logs',     # 日志存储目录
        'result_dir': './pred_results',   # 预测结果目录
        'save_interval': 5,               # 预测结果保存间隔
        }
        
        # 根据输入更新自定义参数
        if custom_params:
            base_params.update(custom_params)
            
        # 转换为类属性（保持点号访问方式）
        for key, value in base_params.items():
            setattr(self, key, value)
        
        # 自动获取可用gpu_ids
        if self.gpu_id is None:  # 自动模式
            try:
                gpu_get = GPUGet()
                self.device_ids = gpu_get.get_available_gpus()
                self.gpu_id = ','.join(map(str, self.device_ids))  # 保持兼容性
            except Exception as e:
                print(f"GPU自动检测失败: {e}, 回退到CPU模式")
                self.device_ids = []
                self.gpu_id = ''
        else:  # 手动指定模式
            self.device_ids = list(map(int, self.gpu_id.split(',')))
            # 验证GPU是否实际可用
            gpu_get = GPUGet()
            available_gpus = gpu_get.get_available_gpus()
            if not set(self.device_ids).issubset(available_gpus):
                raise ValueError(f"指定GPU {self.device_ids} 不可用，当前可用GPU: {available_gpus}")

        # 设备设置（保持多GPU兼容）
        if self.device_ids:
            self.device = torch.device(f"cuda:{self.device_ids[0]}")
            torch.cuda.set_device(self.device)  # 设置默认设备
        else:
            self.device = torch.device("cpu")
            
        # # 动态计算参数（需要动态计算或固定不可被更改的设计常数）
        # self.device = torch.device(f"cuda:{self.gpu_id.split(',')[0]}" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def load(cls, file):
        """从JSON文件加载配置"""
        with open(file, 'r') as f:
            config_data = json.load(f)
            return cls(config_data)

    def __getattr__(self, name):
        """更安全的属性访问"""
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")