import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
from scipy import signal
import os
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, path, input_time_steps,right_trim_time_steps):
        self.path = path
        self.audio_list = os.listdir(self.path)
        
        self.input_time_steps = input_time_steps
        self.right_trim_time_steps = right_trim_time_steps
        
    def __len__(self):
        return len(self.audio_list) #works
    
    def __getitem__(self, idx):
        file_name = self.audio_list[idx]
        label = file_name.split('.')[0].split('_')[-1]
        label = 1 if label=='True' else 0
 
        sample_rate, samples = wavfile.read(os.path.join(self.path, file_name))
        _, _, spectrogram = signal.spectrogram(samples, sample_rate)
        # print(spectrogram.shape)
        
        if label == 0:
            ##if it is false, that random number can be bigger..
            right_trim = np.random.randint(low=1, high=self.right_trim_time_steps*5)
        else:
            right_trim = np.random.randint(low=1, high=self.right_trim_time_steps)

        spectrogram = spectrogram[:,:-right_trim] #this is how many to shave off the end. 
        spec_height,spec_t = spectrogram.shape
        if spec_t < self.input_time_steps:
            pad_width = self.input_time_steps - spec_t 
            pad_zeros = np.zeros((spec_height,pad_width), dtype=np.float32)
            # pad_zeros.shape
            spectrogram = np.concatenate([pad_zeros,spectrogram], axis = 1)
        
        data = torch.from_numpy(spectrogram[:,-self.input_time_steps:]) #the value for time steps is the last x.  20 time steps means last 20
        data = data/data.max()
        return data.unsqueeze(0), label
    


class CustomDatasetRandom(Dataset):
    def __init__(self, path, input_time_steps,right_trim_time_steps, right_trim_time_steps_false):
        self.path = path
        self.audio_list = os.listdir(self.path)
        
        self.input_time_steps = input_time_steps
        self.right_trim_time_steps = right_trim_time_steps
        
    def __len__(self):
        return len(self.audio_list) #works
    
    def __getitem__(self, idx):
        file_name = self.audio_list[idx]
        label = file_name.split('.')[0].split('_')[-1]
        label = 1 if label=='True' else 0
        
        right_trim_time_steps = np.random.randint(right_trim_time_steps)
        ##now right trim randomizes.  keep this small for true samples. 20 max i think
        if label == 0:
            ##if it is false, that random number can be bigger..
            right_trim_time_steps = np.random.randint(right_trim_time_steps_false)
        
        sample_rate, samples = wavfile.read(os.path.join(self.path, file_name))
        _, _, spectrogram = signal.spectrogram(samples, sample_rate)
        # print(spectrogram.shape)
        spectrogram = spectrogram[:,:-self.right_trim_time_steps] #this is how many to shave off the end. 
        spec_height,spec_t = spectrogram.shape
        
        if spec_t < self.input_time_steps:
            pad_width = self.input_time_steps - spec_t 
            pad_zeros = np.zeros((spec_height,pad_width), dtype=np.float32)
            # pad_zeros.shape
            spectrogram = np.concatenate([pad_zeros,spectrogram], axis = 1)
        
        data = torch.from_numpy(spectrogram[:,-self.input_time_steps:]) #the value for time steps is the last x.  20 time steps means last 20
        return data.unsqueeze(0), label
    
    
class CustomDatasetFineTune(Dataset):
    def __init__(self, path, input_time_steps,right_trim_time_steps):
        self.path = path
        self.audio_list = os.listdir(self.path)
        #True.wav, False.wav
        self.input_time_steps = input_time_steps
        self.right_trim_time_steps = right_trim_time_steps
        
    def __len__(self):
        return len(self.audio_list) #works
    
    def __getitem__(self, idx):
        
        file_name = self.audio_list[idx] #has 0 and 1
        label = file_name.split('.')[0]
        
        label = 1 if label=='True' else 0
        
 
        sample_rate, samples = wavfile.read(os.path.join(self.path, file_name))
        _, _, spectrogram = signal.spectrogram(samples, sample_rate)
        
      
        spec_height,spec_t = spectrogram.shape
        trim_idx = np.randint(low = 0, high = spec_t - self.input_time_steps)
        
        if spec_t < self.input_time_steps:
            pad_width = self.input_time_steps - spec_t 
            pad_zeros = np.zeros((spec_height,pad_width), dtype=np.float32)
            # pad_zeros.shape
            spectrogram = np.concatenate([pad_zeros,spectrogram], axis = 1)
        
        data = torch.from_numpy(spectrogram[:,trim_idx:trim_idx+self.input_time_steps]) #the value for time steps is the last x.  20 time steps means last 20
        data = data/data.max()
        return data.unsqueeze(0), label
    

    

    
def get_dataloader(path, batch_size, num_workers, input_time_steps,right_trim_time_steps):
    dataset = CustomDataset(path,input_time_steps,right_trim_time_steps)
    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers = num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers = num_workers, shuffle=True)

    return train_dataloader, val_dataloader