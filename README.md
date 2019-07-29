## Sentiment-Analysis-with-Word-Embeddings
A Sentiment Analysis model in keras to analyse toxic comments online . The Corpus is preprocessed using Glove Word Embeddings .

## Data :

The Corpus used in the following code is taken from a Kaggle competetion "Toxic Comment Classification" . It consists of online 
comments from Wikipedia’s talk page edits .The main task is to classify the comments into different categories such as toxic ,severe toxic ,obscene ,threat ,insult and identity hate .The link to the competetion : https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge (The competetion deadline was over before I started with Sentiment Analysis and thus couldn't submit my result)  



## Word Embeddings :

The Corpus as we all know is raw text and thus needs to be preprocessed before feeding to our model . I used Glove pretrained Word Embeddings to form word vectors of the given raw data . The Word Vectors formed have 100 dimensions . i.e each word was been represented as a 100 dimensional vector .
The link to the pretrained Glove Word Embedding weights : nlp.stanford.edu/data/glove.6B.zip



## Visualizing the Word Embeddings ( Unsupervised Learning Approach ):

We need to visualize our word embedding vectors before using them to see whether related words are positioned close to each other .
As our word vectors have 100 dimensions it is not easy to visualize them . Thus , I used an unsupervised approach (Dimensionality Reduction ) to reduce the dimensions to 2 and then plot these words . The algorithm used is t-SNE(Distributed Stochastic Neighbor Embedding ) . As we have used pretrained weights we get very good results on visualizing the data . This approach is important when we train our own Embeddings .



Finally a Bidirectional LSTM model is built using keras to classify the sentences into appropriate categories .
