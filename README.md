## Social Media Text Processing
#### A model that classifies tweets into 2 classes (informative/non-informative) in context of Covid-19

### About

During the pandemic(COVID19), people post a large number of  informative and non-informative tweets on Twitter. Informative  tweets provide helpful information such as availability and resource  requirements, affected people etc. This information is very helpful for  Government / Organisations. So, the main task of this project is to classify the tweets into  informative and non-informative



### Dataset

- 7000 tweets which were posted on Twitter were used in this  project.
-  3104 tweets of this dataset were manually annotated and remaining tweets gets self annotated in the  Self-Training part

### Topic Modelling

- Representing a document by relying only on its particular  words is not good enough for the efficient comparison of  documents. we need to represent the documents within a  common semantic space. The most essential techniques for  such a representation is Latent Dirichlet Allocation(LDA)
- LDA is a generative probabilistic model that assumes each  topic is a mixture over an underlying set of words, and each  document is a mixture of over a set of topic probabilities
- I considered and tried different values for these parameters for Topic  Modelling and these gave me better results (Coherence score)
- Number of topics
  - Plotted wide range  of number of topics  and their  corresponding  coherence scores  and got a peak at  70. So, I considered  70 topics for LDA
- Number of passes
  - Plotted number of  passes vs  corresponding  coherence scores  and got better  results when this  value is 50. So I  considered 50  passes for my LDA  model.
- Alpha (α)
  - Griffiths and  Steyvers (2004)  suggest a value 0f  50/(number of  topics) for α. So, I  considered 0.694 as  α.

### Semi Supervised Learning

● The Semi-Supervised Learning is an extension of supervised  learning. 

● It provides the benefits of both unsupervised and supervised  learning.  

● It uses a small amount of labelled data and a large amount of  unlabelled data.  

● There are various Semi Supervised Learning  approaches/methods.  

● The most simple and adaptable method is Self Training.



### Self Training

● Self-training is one of the semi-supervised learning methods  that alternatively repeat training a base classifier and labeling  unlabeled data in training set. 

● Here, we generally adapt some confidence measures to label  the data confidently. 

 ● This is because more confident label implies less error  probability.



### Confidence Measure and Threshold

● If the base classifier of Self training algorithm mislabels some  examples and the mislabelled examples gets included in the  data set, then the next classifier may learn improper  classification boundaries and gives us incorrect results. 

● So, to minimise this error, we assign a threshold value to the  base classifier. Its is 0.75 in this project. 

● Only the examples that has prediction score more than this  threshold value gets included in the data set. 

● This is one of the ways we can reduce the error probability at  the initial classification.



### Algorithm for Self Training

Now for training, I converted all the text documents of labelled dataset  into vectors(topic probabilities) and grouped it as X and their  corresponding labels as Y. To train the model, I divided this data set in  75:25 ratio.

- I divided the unlabelled text into 5 groups of similar size. So  that whole data can be trained as separate batches.
- Feed the base classification model with the training data X  train and Y train
- Converted unlabelled text to vectors with topic probabilities  (LDA model) and predict the probability of classification with  the trained base classifier. 
- If this prediction score is greater than threshold or less than  1-threshold, the text and the predicted label gets included in  the main data set. 
-  Repeat the above process for 5 groups of unlabelled data.
-  Return the final labelled data.

### Improving Self Training

In order to get good labelling of the data set, it is very important that  Self Training Algorithm to perform well and train the base classifier  with maximum possible confidence. Now, we have to improve this base Classifier. So, we consider few classification models from Scikit learn Library  and test these classifiers considering various metrics .

### Deep Learning based text  classification

● Deep learning based text classification approaches require word  embeddings (mostly word2vec), but the success of this process relies  on the fact that whether we have access to significant volume of  textual data. 

● To further increase the representation flexibility of deep learning  models, attention mechanisms have been introduced and  developed, which form an integral part of the text classification  process.

● So, I used a deep learning model consists of an encoder and  decoder that are connected through attention  mechanism(modified BERT).

### Bidirectional Encoder Representations  from Transformers (BERT) 

● I developed an improved deep learning-based approach to  automatically classify the text using modified BERT(RoBERTa) BERT  and other Transformer encoder architectures have been wildly  successful on a variety of tasks in NLP 

● The BERT family of models uses the Transformer encoder  architecture to process each token of input text in the full context of  all tokens before and after, hence the name is Bidirectional Encoder  Representations from Transformers.

### About the Model

● Model used : bert-base-uncased 

● Optimizer : Adam Optimizer 

● Loss function : SparseCategoricalCrossentropy 

● Learning rate : 2e-5 

● epochs : 100















