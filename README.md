# DL_STFU
Real time audio censoring for BU Deep Learning Spring 2023

# Dallin:
pipeline_trial.ipynb is a proof of concept for the data pipeline.  it takes text, passes it to the google text to speach api, and turns that into a spectrogram in numpy, which can then be fed to a neural network.

4/2 pipe_line trial.ipynb now has the custom data loader, takes the audio files and some hyperparameters and produces a pytorch dataloader that provides batched and labelled data.

kaggle_data_cleaning.ipynb takes the Dota chat dataset https://www.kaggle.com/datasets/romovpa/gosuai-dota-2-game-chats augments and labels it for our purpose.  cuss_words is a list we maintain who's content is offensive by nature.

labeled_augmented.csv is the current (as of 3/25/23) text data set. Column 0 is the text input, and column 1 is True if the final word contains a member of the cuss_words list and False otherwise.
