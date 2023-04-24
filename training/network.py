import torch
import torch.nn as nn

class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)

        # Linear classifier
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    
class TallSkinny(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(129, 1), stride = 1, padding= 0) #dallin thinks more channels here is better.
        self.relu1 = nn.ReLU() #debate me 
        self.bn1 = nn.BatchNorm2d(16) #don't pool

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), stride= 1, padding= 0)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride= 1, padding= 0)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(1, 3), stride= 1, padding= 0)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(128)
        
        ##linear layers?
        
        #self.conv5 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(2, 2), padding=(1, 1))
        #self.relu5 = nn.ReLU()
        #self.bn5 = nn.BatchNorm2d(64)
        
        #self.conv6 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(2, 2), padding=(1, 1))
        #self.relu6 = nn.ReLU()
        #self.bn6 = nn.BatchNorm2d(64)

        # Linear classifier
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    
class TallSkinny_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1a = nn.Conv2d(1, 4, kernel_size=(129, 1), stride = 1 ) #we think we only left pad, these need to
        self.conv1b = nn.Conv2d(1, 4, kernel_size=(129, 3), stride = 1) #be 
        self.conv1c = nn.Conv2d(1, 4, kernel_size=(129, 5), stride = 1)
        self.conv1d = nn.Conv2d(1, 4, kernel_size=(129, 10), stride = 1)
        
        self.relu1 = nn.ReLU() #debate me 
        self.bn1 = nn.BatchNorm2d(16) #don't pool

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), stride= 1, padding= 0)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride= 1, padding= 0)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(1, 3), stride= 1, padding= 0)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(128)
        
        #add linear layers

        # Linear classifier
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        
        a = self.conv1a(x)
        b = self.conv1b(x)
        b = nn.functional.pad(b,(2,0),value = 0) #these numbers depend on the size of the convolution
        
        c = self.conv1c(x)
        c = nn.functional.pad(c,(4,0),value = 0)
        
        d = self.conv1d(x)
        d = nn.functional.pad(d,(9,0),value = 0)
    
        x = torch.cat((a, b, c, d), 1) #concat over channnel.
        
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
