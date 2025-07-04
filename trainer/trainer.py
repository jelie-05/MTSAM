import os
import torch.nn as nn
from typing import Optional, Union, Dict
from torch.utils.data import DataLoader
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from utility.distributed import is_main_process, get_rank, get_world_size
from pathlib import Path
from datetime import datetime
from utility.distributed import (
    is_distributed, 
    get_rank, 
    get_world_size, 
    is_main_process,
    reduce_dict,
    save_on_master,
    synchronize,
    gather_tensor
)
from utility.metrics import AverageMeter, ConfusionMatrix
from segment_anything.utils.transforms import ResizeLongestSide


class TrainerBDD100K:
    """Trainer class for handling the training loop with distributed support."""

    def __init__(
            self,
            model: nn.Module,
            config,
            device: Optional[torch.device] = None,
            run_id: Optional[str] = None,
            train_loader: Optional[DataLoader] = None,
            val_loader: Optional[DataLoader] = None,
            test_loader: Optional[DataLoader] = None,
            rank: int = 0,
            world_size: int = 1,
            resume_from: Optional[Union[str, Path]] = None,
            resume_training: bool = False,            
    ):
        """
        Initialize the Trainer with the model and other necessary components.
        
        Args:
            model (nn.Module): The model to be trained.
            config: Configuration object containing training parameters.
            device (Optional[torch.device]): The device to run the model on. Defaults to None.
            run_id (Optional[str]): Unique identifier for the training run. Defaults to None.
            train_loader (Optional[DataLoader]): DataLoader for the training dataset. Defaults to None.
            val_loader (Optional[DataLoader]): DataLoader for the validation dataset. Defaults to None.
            test_loader (Optional[DataLoader]): DataLoader for the test dataset. Defaults to None.
            rank (int): The rank of the current process in distributed training. Defaults to 0.
            world_size (int): Total number of processes in distributed training. Defaults to 1.
        """
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.rank = rank
        self.world_size = world_size

        # Set the device for the model
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.use_amp = getattr(self.config, 'use_amp', False)
        self.scaler = GradScaler() if self.use_amp and torch.cuda.is_available() else None

        if is_main_process() and self.use_amp:
            print(f"Using Automatic Mixed Precision (AMP) for training.")
            if not torch.cuda.is_available():
                print("Warning: AMP is enabled but CUDA is not available. Disabling AMP.")
                self.use_amp = False
                self.scaler = None
        
        # Create base experiment directory
        if is_main_process():
            self.experiment_dir = Path(f"outputs/{self.config.experiment}")
            os.makedirs(self.experiment_dir, exist_ok=True)
        else:
            self.experiment_dir = Path(f"outputs/{self.config.experiment}")

        # Handling resuming training
        self.start_epoch = 0
        self.resume_ckpt = None
        
        if resume_from or resume_training:
            # TODO: Implement logic to load the checkpoint and resume training
            pass

        # Generate a unique run ID if not provided
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{self.config.experiment}_{timestamp}"
            if is_main_process():
                print(f"run_id is not defined. Generated run ID: {run_id}")

        # Create run-specific directory
        if is_main_process():
            self.run_dir = self.experiment_dir / f"run_{run_id}"
            if not self.resume_checkpoint and self.output_dir.exists():
                print(f"Warning: Run directory {self.output_dir} already exists. Adding unique suffix.")
                import uuid
                self.output_dir = self.experiment_dir / f"run_{run_id}__{uuid.uuid4().hex[:6]}"
            
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Saving outputs to {self.output_dir}")
        else:
            self.output_dir = self.experiment_dir / f"run_{run_id}"

        # Sync all processes before proceeding
        if is_distributed():
            synchronize()

        if self.train_loader is not None:
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
        else:
            self.optimizer = None
            self.scheduler = None

        # Initialize training state variables
        self.best_val_loss = float('inf')
        self.best_miou = 0.0
        self.best_epoch = 0
        self.best_metrics = {}
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'mean_iou': [],
            'pixel_accuracy': [],
            'learning_rate': []
        }

        # Resume from checkpoint if specified
        if self.resume_ckpt:
            self._load_checkpoint(self.resume_ckpt)
        else:
            if is_main_process():
                self.save_run_config(config)

        # Initialize tensorboard
        self.writer = None

        if is_main_process():
            tensorboard_dir = self.output_dir / 'tensorboard'
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

            print(f"TensorBoard logs will be saved to {tensorboard_dir}")
            print(f"Start TensorBoard with: tensorboard --logdir {tensorboard_dir}")

        # SAM model specific initialization
        self.resize_transform = ResizeLongestSide(model.module.image_encoder.img_size)

    
    def train(self):
        """
        Main training loop.
        """

        if is_main_process():
            print(f"Starting training from epoch {self.start_epoch + 1}")
        
        start_time = time.time()

        for epoch in range(self.start_epoch, self.config.epochs):
            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            # Training step
            train_metrics = self.train_epoch(epoch)

            # Only update history on main process
            if is_main_process():
                self.metrics_history['train_loss'].append(train_metrics['train_loss'])

                # Current learning rate
                lr = self.optimizer.param_groups[0]['lr']
                self.metrics_history['learning_rate'].append(lr)

            # Validation step
            if self.val_loader is not None:
                val_metrics = self.validate_epoch(epoch)

                if is_main_process():
                    self.metrics_history['val_loss'].append(val_metrics.get('val_loss', 0))
                    self.metrics_history['mean_iou'].append(val_metrics.get('mean_iou', 0))
                    self.metrics_history['pixel_accuracy'].append(val_metrics.get('pixel_accuracy', 0))

                    # Check for the best model (based on mIoU)
                    current_miou = val_metrics.get('mean_iou', 0)
                    if current_miou > self.best_miou:
                        self.best_miou = current_miou
                        self.best_val_loss = val_metrics.get('val_loss', float('inf'))
                        self.best_epoch = epoch
                        self.best_metrics = val_metrics.copy()

                        # Save best model checkpoint with metrics
                        self.save_checkpoint(epoch, is_best=True, metrics=val_metrics)

                    # Also keep track of best loss
                    if val_metrics.get('val_loss', float('inf')) < self.best_val_loss and epoch != self.best_epoch:
                        self.best_val_loss = val_metrics.get('val_loss', float('inf'))

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('val_loss', 0))
                else:
                    self.scheduler.step()

            # Save regular checkpoint every 10 epoch or at the end of training
            if (epoch + 1) % 10 == 0 or epoch == self.config.epochs - 1:
                self.save_checkpoint(epoch, metrics=train_metrics)

            if is_main_process():
                lr = self.optimizer.param_groups[0]['lr']
                val_loss_str = f"{val_metrics.get('val_loss', 'N/A'):.4f}" if 'val_loss' in val_metrics else 'N/A'
                miou_str = f"{val_metrics.get('mean_iou', 'N/A'):.4f}" if 'mean_iou' in val_metrics else 'N/A'
                print(f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                      f"Train loss: {train_metrics['train_loss']:.4f}, "
                      f"Val loss: {val_loss_str}, "
                      f"mIoU: {miou_str}, "
                      f"LR: {lr:.6f}")
        
        # Save metrics history plot
        if is_main_process():
            self._save_metrics_plot(self.metrics_history)
        
        # Calculate training time
        total_time = time.time() - start_time
        
        # Return final statistics
        return {
            'total_epochs': self.config.training.epochs,
            'resumed_from_epoch': self.start_epoch + 1,
            'best_val_loss': self.best_val_loss,
            'best_miou': self.best_miou,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'training_time': total_time,
            'metrics_history': self.metrics_history
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Training loop for a single epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
        """
        self.model.train()

        loss_meter = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        cuda_mem = AverageMeter()

        start_time = time.time()

        # Only show progress bar on main process
        if is_main_process():
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
        else:
            pbar = self.train_loader
        
        total_batches = len(self.train_loader)



        for batch_idx, batch in enumerate(pbar):
            # Measure data loading time
            data_time.update(time.time() - start_time)

            # Move batch to device
            if isinstance(batch, dict):
                # TODO: Adjust this based on BDD100K dataset structure (it is from NYU dataset)
                inputs = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                depths = batch.get('depth', None)
                normals = batch.get('normal', None)
            else:
                inputs, labels, depths, normals = batch
                inputs = inputs.to(self.device)
                labels = labels.long().to(self.device)
                depths = depths.to(self.device)
                normals = normals.to(self.device) 

            self.optimizer.zero_grad()

            task_idxs = [_ for _ in range(self.config.task_num)] if self.config.task_num > 1 else 0

            train_pred = []

            for task_idx in task_idxs:
                batched_input = []
                resized_train_data = self.resize_transform.apply_image_torch(inputs)
                for i in range(self.config.batch_size):
                    batched_input.append({'image': resized_train_data[i], 'original_size': inputs[i].shape[1:3]})
                
                batch_pred = self.model(batched_input, task_idx=task_idx, task_slices=self.config.task_slices)
        

    def _load_checkpoint(self, checkpoint_path: Union[str, Path]):
        pass

    def save_run_config(self, config):
        pass
        
    def _create_optimizer():
        pass
        
    def _create_scheduler():
        pass

    def _save_metrics_plot(self, metrics_history):
        pass

    def save_checkpoint(self, epoch: int, is_best: bool = False, metrics: Optional[dict] = None):
        pass


        