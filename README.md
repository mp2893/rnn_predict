# Heart Failure Prediction using RNN
This is a simple RNN (implemented with Gated Recurrent Units) for predicting a HF diagnosis given patient records.
There are four different versions.

1. gru_onehot.py: This uses one-hot encoding for the medical code embedding
2. gru_onehot_time.py: This uses one-hot encoding for the medical code embedding. This uses time information in addition to the code sequences
3. gru_emb.py: This uses pre-trained medical code embeddings. 
4. gru_emb_time.py: This uses pre-trained medical code embeddings. This suses time information in addition to the code sequences.

The data are synthetic and make no sense at all. It is intended only for testing the codes.

1. sequences.pkl: This is a pickled list of list of integers. Each integer is assumed to be some medical code.
2. times.pkl: This is a pickled list of list of integers. Each integer is assumed to the time at which the medical code occurred.
3. labels.pkl: This is a pickled list of 0 and 1s.
4. emb.pkl: This is a randomly generated code embedding of size 100 X 100

# Requirement
Python and Theano are required to run the scripts

# How to Execute
1. python gru_onehot.py sequences.pkl labels.pkl <output>
2. python gru_onehot_time.py sequences.pkl times.pkl labels.pkl <output>
3. python gru_emb.py sequences.pkl labels.pkl emb.pkl <output>
4. python gru_emb_time.py sequences.pkl times.pkl labels.pkl emb.pkl <output>

All scripts will divide the data into training set, validation set, and test set. They will run for a fixed number of epochs. At each epoch, "Validation AUC" will be calculated using the validation set, and if it is the best "Validation AUC" so far, the test set will be used to calculate "Test AUC". The model with the best "Test AUC" will be saved at the end of the training.
