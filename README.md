# DL_STFU
Real time audio censoring for BU Deep Learning Spring 2023

Dataset: [Drive](https://drive.google.com/drive/folders/1fCVkjPTNFxRuT2trkHi7Q7p0KZ-Uv0DY)

# Dallin:
pipeline_trial.ipynb is a proof of concept for the data pipeline.  it takes text, passes it to the google text to speach api, and turns that into a spectrogram in numpy, which can then be fed to a neural network.

4/2 pipe_line trial.ipynb now has the custom data loader, takes the audio files and some hyperparameters and produces a pytorch dataloader that provides batched and labelled data.

kaggle_data_cleaning.ipynb takes the Dota chat dataset https://www.kaggle.com/datasets/romovpa/gosuai-dota-2-game-chats augments and labels it for our purpose.  cuss_words is a list we maintain who's content is offensive by nature.

labeled_augmented.csv is the current (as of 3/25/23) text data set. Column 0 is the text input, and column 1 is True if the final word contains a member of the cuss_words list and False otherwise.



# Himanshu
03/12: Literature survey [Paper](https://ieeexplore.ieee.org/document/9900333)

04/02: Build a network and training pipeline for model training, contained in the training directory

# Nick:
4/1- Have built out the threadpool to handle multithreading for the text to speech API calls. Each row of labeled_aumented.csv is submitted as a job to the thread pool, where the text is submitted to the Google Text-to-Speech API to generate a .wav file.

4/3- Ran the multithreading to generate 212,000 .wav files. Since Colab provides only 2 vCPUs, I was only able to run 2 threads so this took a long time (~5 requests per second x 60 seconds per minute x 60 minutes per hour = ~18,000 results per hour). Next time would be worth running on the SCC to get a machine that would be able to use 32 threads.

# Krishna Adithya:
Literature Survey: 
1. Low-resource Low-footprint Wake-word Detection using Knowledge Distillation
 [Paper](https://arxiv.org/abs/2207.03331)
2. Wake Word Detection with Streaming Transformers [Paper](https://arxiv.org/abs/2102.04488)

4/3: Built a simple rnn architecture, so that we can compare the cnn to rnn model.

4/5: Modified the existing scripts to import rnn model and created a jupyter notebook with training pipeline. 
