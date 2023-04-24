import argparse
import torch
from training import train, validate
import pandas as pd
from dataloader import get_dataloader
from network import AudioClassifier, TallSkinny, TallSkinny_2
import sys
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import wandb

# sys.path.append('/home/himanshu/STFU/')
# print(sys.path)

def main(args):
    trainloader, valloader = get_dataloader(args.data, args.batch_size, args.n_workers, args.input_time_steps, args.right_trim_time_steps)

    if args.model == 'Normal':
        model = AudioClassifier().to(args.device)
    elif args.model == 'TallSkinny':
        model = TallSkinny().to(args.device)
    elif args.model == 'TallSkinny-v2':
        model = TallSkinny_2().to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    for epoch in range(1, args.n_epochs+1):
        args.ep = epoch
        loss = train(args, model, trainloader, criterion, optimizer)
        acc = validate(args, model, valloader)
        wandb.log({'Loss': loss, 'Accuracy': acc}, step = args.ep)
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/'+f'{args.identifier}.pt')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STFU model training')
    parser.add_argument('--data', type=str, default='Data', help='Data Directory')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--n_workers', type=int, default=16, help='Number of workers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
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
    parser.add_argument('--identifier', type=str, default='tall_skinny', help='Identifier for wandb')
    parser.add_argument('--wandb', type=int, default=0, help='Use wandb')
    parser.add_argument('--model', type=str, choices=['Normal', 'TallSkinny', 'TallSkinny-v2'])
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    mode = 'online' if args.wandb else 'disabled'
    wandb.init(project="STFU", name = f'{args.identifier}', reinit=True, config=args, mode=mode)
    main(args)