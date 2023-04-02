import argparse
import torch
from training import train

def main(args):
    
    train(model, trainloader, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STFU model training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_model', action='store_true', default=False, help='Save model')
    parser.add_argument('--save_model_path', type=str, default='model.pt', help='Path to save model')
    parser.add_argument('--save_model_interval', type=int, default=10, help='Save model interval')
    parser.add_argument('--save_model_dir', type=str, default='models', help='Directory to save model')
    parser.add_argument('--save_model_name', type=str, default='model', help='Name to save model')
    
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)