# Automatic Detection of Sexist Statements Commonly Used at the Workplace

Repository for "Automatic Detection of Sexist Statements Commonly Used at the Workplace" by Dylan Grosz & Patricia Conde-Cespedes. The associated paper was accepted to the PAKDD Workshop on Learning Data Representation for Clustering (LDRC 2020).

Associated Paper: https://pakdd2020.org/download/workshop_paper/LDRC2020_2.pdf

## Abstract
> Detecting hate speech in the workplace is a unique classification task, as the underlying social context implies a subtler version of conventional hate speech. Applications regarding a state-of-the-art workplace sexism detection model include aids for Human Resources departments, AI chatbots and sentiment analysis. Most existing hate speech detection methods, although robust and accurate, focus on hate speech found on social media, specifically Twitter. The context of social media is much more anonymous than the workplace, therefore it tends to lend itself to more aggressive and “hostile” versions of sexism. Therefore, datasets with large amounts of “hostile” sexism have a slightly easier detection task since “hostile” sexist statements can hinge on a couple words that, regardless of context, tip the model off that a statement is sexist. In this paper we present a dataset of sexist statements that are more likely to be said in the workplace as well as a deep learning model that can achieve state-of-the art results. Previous research has created state-of-the-art models to distinguish “hostile” and “benevolent” sexism based simply on aggregated Twitter data. Our deep learning methods, initialized with GloVe or random word embeddings, use LSTMs with attention mechanisms to outperform those models on a more diverse, filtered dataset that is more targeted towards workplace sexism, leading to an F1 score of 0.88.

## Getting Started

### Environment

After cloning the repository, access the resulting folder and run ```pip install -r requirements.txt```. It is recommended to install all of the packages in ```requirements.txt```. However, some of these packages may be superfluous, as most of this paper's development occurred in PyCharm.

### Downloading Dependent Data

#### GloVe Word Embeddings
Download a GloVe dataset from https://nlp.stanford.edu/projects/glove/. For both memory considerations and replication of this research, download the 6 billion token, 50D word vectors http://nlp.stanford.edu/data/glove.6B.zip. 50 dimensions were chosen over larger embedding dimensions in order to prevent overfitting/memorization during training on such a small dataset.

A similarly trained gender balanced GloVe dataset (GN-GloVe) can be downloaded here: https://drive.google.com/file/d/1v82WF43w-lE-vpZd0JC1K8WYZQkTy_ii/view

After downloading the dataset, make sure to either name your embedding text file ```vectors.txt``` after placing it in the ```data``` folder, or edit ```embedding_fn``` in ```SexisteDetectionMain.py``` (line 33) so it points to the correct filepath.

#### Sexist Workplace Statements Dataset

The dataset we used for training is provided in ```data/SD_dataset_FINAL.csv```. It is also present on Kaggle: https://www.kaggle.com/dgrosz/sexist-workplace-statements. If you would like to update the dataset and reupload, either place it in the ```data``` folder and rename it to ```SD_dataset_FINAL.csv``` or edit the ```sexist_dataset_fn``` variable on line 32 of ```SexisteDetectionMain.py```.

### Running the Models

After all the embeddings and sexist workplace statements datasets are properly loaded and referenced, run ```SexisteDetectionMain.py```. There are 8 versions of the model that we have coded, each with their own architecture:
 - ModelV1: word embeddings are concatenated per phrase, averaged and put through logistic regression
 - ModelV2: word embeddings are sequentially pushed through a Vanilla Unidirectional LSTM architecture, specifically ```Embedding->LSTM->Dropout->LSTM->FullyConnected->Softmax```
 - ModelV3: word embeddings (you can toggle between random embeddings and loaded GloVe embeddings in the function call) are sequentially pushed through a Bidirectional LSTM architecture, specifically ```Embedding->BiLSTM->Dropout->BiLSTM->FullyConnected->Softmax```
 - ModelV4: embeddings (you can toggle between random embeddings and loaded GloVe embeddings in the function call) are pushed through a Bidirectional LSTM architecture with attention, specifically ```Embedding->BiLSTM->Dropout->BiLSTM->Attention->FullyConnected->Softmax```
 
If you don't change the initial code and run ```SexisteDetectionMain.py```, all of these listed models will run sequentially, performing 10 iterations of training and testing (respective training/testing sets are chosen at random for each iteration). You can edit number of iterations on line 330, and you can edit the list of which models to run in the runModels boolean list (lines 337-341).

For any questions, please email Dylan Grosz at _dgrosz 'at' stanford 'dot' edu_.
