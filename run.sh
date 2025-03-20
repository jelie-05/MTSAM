td_type=tucker
R1=32
R2=32
R3=4
lr=1e-3
lambda_=1
scheduler=linear
if_q=True
if_k=True
if_v=True
warmup_step_ratio=0.05
dropout_rate=0.1
scaling=0.5
batch_size=1
accumulate_step=1
gpu_id=0,1,2,3
nproc_per_node=4
total_epoch=200
data_root=
python -m torch.distributed.launch --nproc_per_node ${nproc_per_node} --master_port 29501 train_nyu.py --gpu_id ${gpu_id} --data_root ${data_root}\
                                    --td_type ${td_type} --R1 ${R1} --R2 ${R2} --R3 ${R3} --enable_qkv ${if_q} ${if_k} ${if_v} --scaling ${scaling} --dropout_rate ${dropout_rate}\
                                    --lr ${lr} --warmup_step_ratio ${warmup_step_ratio} --total_epoch 200 --scheduler ${scheduler} --lambda_ ${lambda_}\
                                    --batch_size ${batch_size} --accumulate_step ${accumulate_step}