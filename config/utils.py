import argparse
import os

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model with flexible datasets.")
    
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file.')
    parser.add_argument('--experiment', type=str, default='bdd100k', help='Name of the experiment or dataset to use.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--run_id', type=str, default=None, help='Run ID for tracking experiments.')

    # SAM arguments
    parser.add_argument('--sam_checkpoint', default='checkpoints/sam_vit_l_0b3195.pth', type=str)
    parser.add_argument('--model_type', default='vit_l', type=str)

    # Add arguments for distributed training
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model.')
    

    # Add more arguments as needed
    args = parser.parse_args()
    
    return args

def load_config_from_args():
    args = parse_args()