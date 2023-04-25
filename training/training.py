import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

def train(args, model, trainloader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    tq = tqdm(trainloader, position=0, leave=True, ascii=True)
    for i, data in enumerate(tq):
        # get the inputs and labels and put them on the GPU
        inputs, labels = data[0].to(args.device), data[1].to(args.device)
        assert inputs.max() <= 1.0
        # ANY Prepocessing of input and labels goes here
        labels = labels.float()

        output = model(inputs)

        # print(output, labels.shape)
        loss = criterion(torch.squeeze(output), labels)
        tq.set_description("L %.4f" %loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch: {args.ep}, Loss: {running_loss/len(trainloader)}')
    return running_loss/len(trainloader)


def validate(args, model, valloader):
    model.eval()
    start_test = True
    with torch.no_grad():
        for i, data in enumerate(valloader):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            labels = labels.float().unsqueeze(1)
            output = model(inputs)
            if start_test:
                all_output = torch.sigmoid(output).float()
                all_label = labels.squeeze().float()
                start_test = False
            else:
                all_output = torch.cat((all_output, torch.sigmoid(output).float()), 0)
                all_label = torch.cat((all_label, labels.squeeze().float()), 0)

    predicted = (all_output > 0.5).float()
    accuracy = 100.0*torch.sum(torch.squeeze(predicted).float() == all_label).item() / float(all_label.size()[0])

    print(f'Validation Accuracy: {accuracy}')
    return accuracy
        

