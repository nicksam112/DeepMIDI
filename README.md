# DeepMIDI
### A project for the generation of MIDI files through deep learning.

This project uses Keras | Tensorflow | and [Mido](https://github.com/mido/mido) to train and produce models that can generate single track MIDI music. The MIDI files needs to be parsed before they can be fed into the model, for now I've provided the files necessary to train the model on a small corpus of Bach songs. I'll be cleaning and uploading the parser at a later date. Two types of models are included, a Convolutional model and an LSTM model to try and comapre how well each produces music. They're each trained by giving them a sample of music, and being asked to predict the next note. The LSTM model tends to do better, and I've found an accuracy of 40-50% is typically enough to start producing ok sounding music. 

[Example of a short song produced with the LSTM model](https://drive.google.com/file/d/1LlkFA9h7QEDJRjDFm7C4wr12jYb8JYnU/view?usp=sharing)

I tend to use this site for testing MIDI files as I don't have a MIDI player on my own device: [link](http://midiplayer.ehubsoft.net/)

Flags | Usage
--- | --- 
--run | Runs the model and produces a song. use --save_song_path to give it an output path and --load_weights to load a certain weights file
--train | Train model, be sure to define weights save path with --save_weights
--conv | Use 1D Convolution Model instead of LSTM
--load_weights | Weights file to load
--save_weights | Save weights as
--midi_data_path | Path to midi data
--save_song_path | Path to save produced song