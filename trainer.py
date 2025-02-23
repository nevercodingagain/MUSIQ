import os
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr


""" train model """
def train_epoch(config, epoch, model, criterion, optimizer, scheduler, train_loader):
    losses = []
    model.train()
    
    # 分布式处理
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    
    for data in tqdm(train_loader, desc=f'Epoch {epoch+1}', ncols=150, disable=(rank != 0)):  # 只在主进程显示进度条
        d_img_org = data['d_img_org'].to(config.device, non_blocking=True)
        d_img_scale_1 = data['d_img_scale_1'].to(config.device, non_blocking=True)
        d_img_scale_2 = data['d_img_scale_2'].to(config.device, non_blocking=True)
        labels = data['score'].float().squeeze().to(config.device, non_blocking=True)

        # 动态生成mask_inputs
        real_batch_size = d_img_org.size(0)
        mask_inputs = torch.ones(real_batch_size, config.n_enc_seq+1, device=config.device)

        optimizer.zero_grad()

        # 统一的前向传播
        pred = model(mask_inputs, d_img_org, d_img_scale_1, d_img_scale_2)
        loss = criterion(pred.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 保存结果使用Tensor保持设备一致
        pred_epoch.append(pred.detach().squeeze())
        labels_epoch.append(labels.detach())

    # 跨进程结果聚合
    pred_tensor = torch.cat(pred_epoch)
    labels_tensor = torch.cat(labels_epoch)
    
    # 分配收集结果的内存
    gathered_pred = [torch.zeros_like(pred_tensor) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(world_size)]
    
    dist.all_gather(gathered_pred, pred_tensor)
    dist.all_gather(gathered_labels, labels_tensor)
    
    # 仅在主进程计算结果
    if rank == 0:
        all_pred = torch.cat(gathered_pred).cpu().numpy()
        all_labels = torch.cat(gathered_labels).cpu().numpy()
        rho_s, _ = spearmanr(all_pred, all_labels)
        rho_p, _ = pearsonr(all_pred, all_labels)
        avg_loss = loss.item()  # 获取最后一步的loss值
    else:
        rho_s, rho_p, avg_loss = 0.0, 0.0, 0.0

    # 广播结果到各进程保持同步
    avg_loss = torch.tensor([avg_loss], device=config.device, dtype=torch.float32)
    dist.broadcast(avg_loss, src=0)
    
    rho_s_tensor = torch.tensor([rho_s], device=config.device, dtype=torch.float32)
    dist.broadcast(rho_s_tensor, src=0)
    
    rho_p_tensor = torch.tensor([rho_p], device=config.device, dtype=torch.float32)
    dist.broadcast(rho_p_tensor, src=0)

    # 主进程打印结果
    if rank == 0:
        print(f'[train] epoch:{epoch+1} / loss:{avg_loss.item():.4f} '
              f'/ SROCC:{rho_s_tensor.item():.4f} / PLCC:{rho_p_tensor.item():.4f}')

    return avg_loss.item(), rho_s_tensor.item(), rho_p_tensor.item()

""" validation """
def eval_epoch(config, epoch, model, criterion, test_loader):
    model.eval()  # 统一设置评估模式

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    pred_epoch = []
    labels_epoch = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f'Epoch {epoch+1}', ncols=150, disable=(rank != 0)):
            d_img_org = data['d_img_org'].to(config.device, non_blocking=True)
            d_img_scale_1 = data['d_img_scale_1'].to(config.device, non_blocking=True)
            d_img_scale_2 = data['d_img_scale_2'].to(config.device, non_blocking=True)
            labels = data['score'].float().squeeze().to(config.device, non_blocking=True)

            real_batch_size = d_img_org.size(0)
            mask_inputs = torch.ones(real_batch_size, config.n_enc_seq+1, device=config.device)

            pred = model(mask_inputs, d_img_org, d_img_scale_1, d_img_scale_2)
            loss = criterion(pred.squeeze(), labels)

            pred_epoch.append(pred.detach().squeeze())
            labels_epoch.append(labels.detach())

    # 结果聚合
    pred_tensor = torch.cat(pred_epoch)
    labels_tensor = torch.cat(labels_epoch)
    
    gathered_pred = [torch.zeros_like(pred_tensor) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(world_size)]
    
    dist.all_gather(gathered_pred, pred_tensor)
    dist.all_gather(gathered_labels, labels_tensor)

    # 主进程计算指标
    if rank == 0:
        all_pred = torch.cat(gathered_pred).cpu().numpy()
        all_labels = torch.cat(gathered_labels).cpu().numpy()
        rho_s, _ = spearmanr(all_pred, all_labels)
        rho_p, _ = pearsonr(all_pred, all_labels)
        avg_loss = loss.item()
    else:
        rho_s, rho_p, avg_loss = 0, 0, 0

    # 维持各进程参数同步
    avg_loss = torch.tensor([avg_loss], device=config.device)
    dist.broadcast(avg_loss, src=0)
    
    rho_s_tensor = torch.tensor([rho_s], device=config.device)
    dist.broadcast(rho_s_tensor, src=0)
    
    rho_p_tensor = torch.tensor([rho_p], device=config.device)
    dist.broadcast(rho_p_tensor, src=0)

    if rank == 0:
        print(f'[test] epoch:{epoch+1} / loss:{avg_loss.item():.4f} '
              f'/ SROCC:{rho_s_tensor.item():.4f} / PLCC:{rho_p_tensor.item():.4f}')

    return avg_loss.item(), rho_s_tensor.item(), rho_p_tensor.item()
