import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

def train(model, trainloader, args):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for n_ep, epoch in enumerate(range(args.epochs)):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        tq = tqdm(trainloader, position=0, leave=True, ascii=True)
        for i, data in enumerate(tq):
            # get the inputs and labels and put them on the GPU
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            # ANY Prepocessing of input and labels goes here
            labels = labels.float().unsqueeze(1)

            output = model(inputs)
            # print(output, labels.shape)
            loss = criterion(output, labels)
            tq.set_description("L %.4f" %loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
        print(f'Epoch: {n_ep}, Loss: {running_loss}')

        torch.save(model.state_dict(), 'models/'+'model.pt')

