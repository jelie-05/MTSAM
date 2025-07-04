import signal
import sys
import torch
from torchinfo import summary
from torch.nn.parallel import DistributedDataParallel as DDP
from segment_anything import sam_model_registry
from utility.distributed import (
    setup_distributed, 
    cleanup_distributed, 
    is_main_process,
    get_rank,
    get_world_size,
    init_seeds
)
from config import load_config_from_args, EXPERIMENT_CONFIGS
from segment_anything.modeling.tensorlib import mark_only_td_as_trainable, orthogonal_reg



def signal_handler(signum, frame):
    """Handle termination signals to exit gracefully."""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    cleanup_distributed()
    sys.exit(0)


def main():
    """Main function to train the model for flexible datasets."""
    
    # Register signals for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup distributed training or single GPU/CPU training
    device, rank, world_size, local_rank = setup_distributed()

    try:
        config_yaml, args = load_config_from_args()

        # Initialize random seeds for reproducibility
        init_seeds(args.seed, rank)

        # TODO: Load checkpoint if provided

        # TODO: Dataset and DataLoader setup

        run_id = getattr(args, 'run_id', None)

        config = {} # TODO: to assign

        # task_num from experiment_configs.py

        if args.experiment not in EXPERIMENT_CONFIGS:
            raise ValueError(f"Experiment '{args.experiment}' not found in configurations.")
        task_info = EXPERIMENT_CONFIGS.get(args.experiment, None)
        channel = task_info['channel']
        task_slices = task_info['task_slices']

        sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint, config=config, output_channel=channel).to(device)
        mark_only_td_as_trainable(sam)
        
        # Print model summary if main process
        if is_main_process():
            try:
                summary(sam, depth=8)
                print("Model loaded successfully.")
                trainable_params = sum(p.numel() for p in sam.parameters() if p.requires_grad)
                all_params = sum(p.numel() for p in sam.parameters())
                print(
                    f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.2f}"
                )
                
            except Exception as e:
                print(f"Error printing model summary: {e}")

        # Wrap the model for distributed training
        if world_size > 1:
            sam = DDP(
                sam,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=True,    # Track unused parameters (e.g., frozen layers) and skip them
                gradient_as_bucket_view=True    # Optimize gradient memory layout
            )
            if is_main_process():
                print(f"Distributed training initialized with {world_size} processes.")

        if is_main_process():
            print(f"Training configuration:")
            print(f"- Experiment: {config.experiment}")
            print(f"- Device: {device}")
            print(f"- World size: {world_size}")
            print(f"- Rank: {rank}")
            print(f"- Batch size per GPU: {args.batch_size}")
            print(f"- Effective batch size: {args.batch_size * world_size}")
            print(f"- Learning rate: {args.learning_rate}")
            print(f"- Epochs: {args.epochs}")

    finally:
        # Cleanup distributed training environment
        cleanup_distributed()
