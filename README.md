# (ICLR 2025) MTSAM: Multi-Task Fine-tuning for Segment Anything Model

[Xuehao Wang](https://openreview.net/profile?id=~Xuehao_Wang3), [Zhan Zhuang](https://openreview.net/profile?id=~Zhan_Zhuang1), [Feiyang Ye](https://openreview.net/profile?id=~Feiyang_Ye4), [Yu Zhang](https://openreview.net/profile?id=~Yu_Zhang3)

Official Implementation of ICLR 2025 paper "[MTSAM: Multi-Task Fine-tuning for Segment Anything Model](https://openreview.net/forum?id=6N4QMbeVaO)".

## Abstract

The Segment Anything Model (SAM), with its remarkable zero-shot capability, has the potential to be a foundation model for multi-task learning. However, adopting SAM to multi-task learning faces two challenges: (a) SAM has difficulty generating task-specific outputs with different channel numbers, and (b) how to fine-tune SAM to adapt multiple downstream tasks simultaneously remains unexplored. To address these two challenges, in this paper, we propose the Multi-Task SAM (MTSAM) framework, which enables SAM to work as a foundation model for multi-task learning. MTSAM modifies SAM's architecture by removing the prompt encoder and implementing task-specific no-mask embeddings and mask decoders, enabling the generation of task-specific outputs. Furthermore, we introduce Tensorized low-Rank Adaptation (ToRA) to perform multi-task fine-tuning on SAM. Specifically, ToRA injects an update parameter tensor into each layer of the encoder in SAM and leverages a low-rank tensor decomposition method to incorporate both task-shared and task-specific information. Extensive experiments conducted on benchmark datasets substantiate the efficacy of MTSAM in enhancing the performance of multi-task learning.

## How to run this code

```sh
sh run.sh
```

## Pre-trained Model

We use the checkpoint of SAM-L from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

## Environment

+ Python 3.10.4
+ torch 2.0.0+cu118
+ numpy 1.22.3

## Citation

If you find MTSAM is useful for your research and applications, please cite using this BibTeX:

```b
@inproceedings{
    wang2025mtsam,
    title={{MTSAM}: Multi-Task Fine-Tuning for Segment Anything Model},
    author={Xuehao Wang, Zhan Zhuang, Feiyang Ye, Yu Zhang},
    booktitle={International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=6N4QMbeVaO}
}
```

