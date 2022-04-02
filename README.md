# representation_learning


## Project Description:
This is the work we did for our Representation Learning project 
In this directory you will find 1 video, 4 .py files,  some dataset in pkl format and the three original data sets used to train and test the models.

The main objective of this project is to implement the DialogueRNN model in Tensorflow. 
We trained and tested our implementation with data from the Friends series. The algorithm predicts the emotion of a character when he announces a line. Finally we compared the performance of our algorithm with a simple CNN.

## Implementation, Test \& Comparison 

dialgoue_rnn.py: 
File that allows to train Dialogue RNN, it contains the algorithm implemented in Tensorflow.
To use this file train_data.pkl is required.

simple_cnn.py:
File that allows to train CNN, it contains the algorithm implemented in Tensorflow.
To use this file train_data.pkl is required.

compare_networks.py:
File that allows to test Dialogue RNN and CNN.
To use this file test_data.pkl is required.


utils.py: 
File that contains additional functions to our implementation such as data pre-processing, batch creation and others.
To use this file glove.840B.300d.txt  is required, it can be download at this adress : 
https://www.kaggle.com/datasets/takuok/glove840b300dtxt

---

Credits: This work was made by Lia Furtado and Hugo Vinson. 

Useful links: 
- GIT Repository: https://github.com/liasucf/representation_learning
- Download Glove: https://www.kaggle.com/datasets/takuok/glove840b300dtxt
- Original article : https://arxiv.org/abs/1811.00405

