import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
from scipy import signal

class CustomDataset(Dataset):
    def __init__(self, df, input_time_steps,right_trim_time_steps):
        self.df = df
        self.input_time_steps = input_time_steps
        self.right_trim_time_steps = right_trim_time_steps
        
    def __len__(self):
        return len(self.df) #works
    
    def __getitem__(self, index):
        file_name = self.df.loc[index, 'file_name']
        label = self.df.loc[index, 'label'] #this is binary right now.  
        sample_rate, samples = wavfile.read(file_name)
        _, _, spectrogram = signal.spectrogram(samples, sample_rate)
        print(spectrogram.shape)
        spectrogram = spectrogram[:,:-self.right_trim_time_steps] #this is how many to shave off the end. 
        data = torch.from_numpy(spectrogram[:,-self.input_time_steps:]) #the value for time steps is the last x.  20 time steps means last 20
        return data.unsqueeze(0), label
    
def get_dataloader(df, batch_size,input_time_steps,right_trim_time_steps):
    dataset = CustomDataset(df,input_time_steps,right_trim_time_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader