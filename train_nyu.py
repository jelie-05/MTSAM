import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import numpy as np
import gc

from dataset.nyu.create_dataset import NYUv2
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry
from segment_anything.modeling.tensorlib import mark_only_td_as_trainable, orthogonal_reg
from utils import model_fit, depth_error, normal_error, ConfMatrix, create_optimizer_scheduler
from contextlib import nullcontext

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def parse_args():
    parser = argparse.ArgumentParser(description= 'For NYUv2')
    parser.add_argument('--data_root', default='', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu_id', type=str)
    
    # SAM Argument
    parser.add_argument('--sam_checkpoint', default='checkpoints/sam_vit_l_0b3195.pth', type=str)
    parser.add_argument('--model_type', default='vit_l', type=str)
    
    # Hyper Parameter
    parser.add_argument('--td_type', type=str)
    parser.add_argument('--R1', default=-1, type=int)
    parser.add_argument('--R2', default=-1, type=int)
    parser.add_argument('--R3', default=-1, type=int)
    parser.add_argument('--enable_qkv', metavar='bool', type=bool, nargs='+')
    parser.add_argument('--dropout_rate', default=0.05, type=float)
    parser.add_argument('--scaling', default=0.5, type=float)
    parser.add_argument('--lambda_', default=0.1, type=float)
    
    # Scheduler
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--scheduler', default='linear', type=str)
    parser.add_argument('--warmup_step_ratio', default=0.1, type=float)
    
    # Training Argument
    parser.add_argument('--total_epoch', default=200, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--accumulate_step', default=4, type=int)
    parser.add_argument('--if_mtl_input', default=False, action='store_true')
    
    parser.add_argument('--local-rank', default=-1, type=int)
    return parser.parse_args()

params = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

local_rank = params.local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
device = torch.device("cuda", local_rank)


torch.manual_seed(params.seed)
random.seed(params.seed)
np.random.seed(params.seed)

batch_size = params.batch_size
task_num = 3
nyuv2_train_set = NYUv2(root=params.data_root, mode='trainval', augmentation=True, task='multi-task')
nyuv2_test_set = NYUv2(root=params.data_root, mode='test', augmentation=False, task='multi-task')

train_sampler = torch.utils.data.distributed.DistributedSampler(nyuv2_train_set)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True)

nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
    sampler=train_sampler)

config = {'type': params.td_type, 'R1': params.R1, 'R2': params.R2, 'R3': params.R3, 'scaling': params.scaling,
          'task_num': task_num, 'enable_qkv': params.enable_qkv, 'dropout_rate': params.dropout_rate}

channel = [13, 1, 3]
task_slices = [slice(0, 13), slice(13, 14), slice(14, None)]
    
sam = sam_model_registry[params.model_type](checkpoint=params.sam_checkpoint, config=config, output_channel=channel).to(device)
mark_only_td_as_trainable(sam)
sam = DDP(sam, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    
if dist.get_rank() == 0:
    print_trainable_parameters(sam)

resize_transform = ResizeLongestSide(sam.module.image_encoder.img_size)

trainable_params = []
for n, p in sam.module.named_parameters():
    if p.requires_grad:
        trainable_params.append(p)

total_epoch = params.total_epoch
train_batch = len(nyuv2_train_loader)
params.max_step = train_batch * total_epoch
params.warmup_step = params.max_step * params.warmup_step_ratio

optimizer = optim.Adam(trainable_params, lr=params.lr, weight_decay=1e-6)
scheduler = create_optimizer_scheduler(optimizer, params)

if dist.get_rank() == 0:
    print(params)
    print('LOSS FORMAT: DEPTH_LOSS ABS_ERR REL_ERR')

world_size = torch.distributed.get_world_size()

avg_cost = torch.zeros([total_epoch, 24])
cost = torch.zeros(24)
task_weight = torch.Tensor([1, 1, 4])
for epoch in range(total_epoch):
    s_t = time.time()
    cost = torch.zeros(cost.shape)

    # iteration for all batches
    sam.module.train()
    nyuv2_train_loader.sampler.set_epoch(epoch)
    train_dataset = iter(nyuv2_train_loader)
    
    conf_mat = ConfMatrix(13)
    
    for batch_index in range(train_batch):
        if batch_index % params.accumulate_step == 0:
            optimizer.zero_grad()
            
            task_idxs = [_ for _ in range(task_num)]
        
        train_data, train_label, train_depth, train_normal = next(train_dataset)
        train_data, train_label = train_data.to(device), train_label.long().to(device)
        train_depth, train_normal = train_depth.to(device), train_normal.to(device)
        
        my_context = sam.no_sync if local_rank != -1 and batch_index % params.accumulate_step != 0 else nullcontext
        
        with my_context():
            if params.if_mtl_input:
                batched_input = []
                resized_train_data = resize_transform.apply_image_torch(train_data)
                for i in range(batch_size):
                    batched_input.append({'image': resized_train_data[i], 'original_size': train_data[i].shape[1:3]})
                
                batch_pred = sam(batched_input, task_idx=-1, task_slices=task_slices)
                
                train_pred_seg, train_pred_dep, train_pred_nor = [], [], []
                for i in range(len(batch_pred)):
                    train_pred_seg.append(batch_pred[i]['masks'][0])
                    train_pred_dep.append(batch_pred[i]['masks'][1])
                    train_pred_nor.append(batch_pred[i]['masks'][2])
                    
                train_pred_seg = F.log_softmax(torch.cat(train_pred_seg), dim=1)
                train_pred_dep = torch.cat(train_pred_dep)
                train_pred_nor = torch.cat(train_pred_nor)
                train_pred_nor = train_pred_nor / torch.norm(train_pred_nor, p=2, dim=1, keepdim=True)
                train_pred = [train_pred_seg, train_pred_dep, train_pred_nor]
                
                train_loss = [
                    model_fit(train_pred[0], train_label, 'semantic'),
                    model_fit(train_pred[1], train_depth, 'depth'),
                    model_fit(train_pred[2], train_normal, 'normal'),
                ]
                loss = torch.zeros(task_num).to(device)
                for i in range(task_num):
                    loss[i] = train_loss[i]
                loss = loss * task_weight / torch.sum(task_weight)
                loss = torch.sum(loss) / params.accumulate_step / world_size
            
                loss.backward()
                del loss
            else:
                train_pred = []
                for task_idx in task_idxs:
                    batched_input = []
                    resized_train_data = resize_transform.apply_image_torch(train_data)
                    for i in range(batch_size):
                        batched_input.append({'image': resized_train_data[i], 'original_size': train_data[i].shape[1:3]})
                    
                    batch_pred = sam(batched_input, task_idx=task_idx, task_slices=task_slices)
                    
                    task_pred = []
                    
                    for i in range(batch_size):
                        task_pred.append(batch_pred[i]['masks'][0])
                        
                    task_pred = torch.cat(task_pred)
                    if task_idx == 0:
                        task_pred = F.log_softmax(task_pred, dim=1)
                    if task_idx == 1:
                        task_pred = task_pred
                    if task_idx == 2:
                        task_pred = task_pred / torch.norm(task_pred, p=2, dim=1, keepdim=True)

                    target = [train_label, train_depth, train_normal]
                    task_names = ['semantic', 'depth', 'normal']
                    loss = model_fit(task_pred, target[task_idx], task_names[task_idx]) * task_weight[task_idx] / params.accumulate_step / world_size / torch.sum(task_weight)
                    loss.backward()
                    del loss
        
        if (batch_index + 1) % params.accumulate_step == 0:
            (params.lambda_ * orthogonal_reg(sam, params.td_type, device) / world_size).backward()
            optimizer.step()
            scheduler.step()
        
    # evaluating test data
    if dist.get_rank() == 0:
        sam.module.eval()
        with torch.no_grad():  # operations inside don't track history
            val_dataset = iter(nyuv2_test_loader)
            val_batch = len(nyuv2_test_loader)
            for k in range(val_batch):
                val_data, val_label, val_depth, val_normal = next(val_dataset)
                val_data, val_label = val_data.to(device), val_label.long().to(device)
                val_depth, val_normal = val_depth.to(device), val_normal.to(device)
                
                batched_input = []
                resized_val_data = resize_transform.apply_image_torch(val_data)
                for i in range(val_data.shape[0]):
                    batched_input.append({'image': resized_val_data[i], 'original_size': val_data[i].shape[1:3]})
                batch_pred = sam(batched_input, task_idx=-1, task_slices=task_slices)
                val_pred_seg, val_pred_dep, val_pred_nor= [], [], []
                for i in range(len(batch_pred)):
                    val_pred_seg.append(batch_pred[i]['masks'][0])
                    val_pred_dep.append(batch_pred[i]['masks'][1])
                    val_pred_nor.append(batch_pred[i]['masks'][2])
                val_pred_seg = F.log_softmax(torch.cat(val_pred_seg), dim=1)
                val_pred_dep = torch.cat(val_pred_dep)
                val_pred_nor = torch.cat(val_pred_nor)
                val_pred_nor = val_pred_nor / torch.norm(val_pred_nor, p=2, dim=1, keepdim=True)
                
                val_loss = [
                    model_fit(val_pred_seg, val_label, 'semantic'),
                    model_fit(val_pred_dep, val_depth, 'depth'),
                    model_fit(val_pred_nor, val_normal, 'normal'),
                ]
                
                conf_mat.update(val_pred_seg.argmax(1).flatten(), val_label.flatten())
                cost[12] = val_loss[0].item()
                cost[15] = val_loss[1].item()
                cost[16], cost[17] = depth_error(val_pred_dep, val_depth)
                cost[18] = val_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(val_pred_nor, val_normal)
                avg_cost[epoch, 12:] += cost[12:] / val_batch
                
            avg_cost[epoch, 13], avg_cost[epoch, 14] = conf_mat.get_metrics()
        
        e_t = time.time()
        
        print('Epoch: {:04d} | TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} || {:.4f}'
                .format(epoch,  avg_cost[epoch, 12], avg_cost[epoch, 13],
                        avg_cost[epoch, 14], avg_cost[epoch, 15], avg_cost[epoch, 16], avg_cost[epoch, 17], avg_cost[epoch, 18],
                        avg_cost[epoch, 19], avg_cost[epoch, 20], avg_cost[epoch, 21], avg_cost[epoch, 22], avg_cost[epoch, 23], e_t-s_t))
    
    torch.cuda.empty_cache()
