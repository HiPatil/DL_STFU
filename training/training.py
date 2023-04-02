import torch
import torch.nn as nn
import torch.optim as optim

def train(model, trainloader, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(args.epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs and labels and put them on the GPU
            inputs, labels = data[0].to(args.device), data[1].to(args.device)

            # ANY Prepocessing of input and labels goes here

            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
