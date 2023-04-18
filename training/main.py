import argparse
import torch
from training import train
import pandas as pd
from dataloader import get_dataloader
from network import AudioClassifier
import sys
sys.path.append('/home/himanshu/STFU/')
# print(sys.path)

def main(args):
    trainloader = get_dataloader(args.data, args.batch_size, args.n_workers, args.input_time_steps, args.right_trim_time_steps)

    model = AudioClassifier().to(args.device)

    train(model, trainloader, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STFU model training')
    parser.add_argument('--data', type=str, default='Data', help='Data Directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--n_workers', type=int, default=16, help='Number of workers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_model', action='store_true', default=False, help='Save model')
    parser.add_argument('--save_model_path', type=str, default='model.pt', help='Path to save model')
    parser.add_argument('--save_model_interval', type=int, default=10, help='Save model interval')
    parser.add_argument('--save_model_dir', type=str, default='models', help='Directory to save model')
    parser.add_argument('--save_model_name', type=str, default='model', help='Name to save model')
    parser.add_argument('--input_time_steps', type=int, default=100, help='Number of time steps to use as input')
    parser.add_argument('--right_trim_time_steps', type=int, default=1, help='Number of time steps to trim off the end of the spectrogram')
    
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)