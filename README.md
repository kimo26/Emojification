
# Emojification
I made this notebook for when I was giving a workshop in collaboration with Google Cloud at the AI+X Summit 2022 in Zurich. This was meant to introduce NLP problems in Deep Learning to the audience. The example we used was a bot that recommends emojis based on the user's input sentence. 
# Introduction
The ACE workshop is built around the idea that the best way to learn something new is by getting a hands-on experience. Leanring something new is a journey form where you are now to where you want to be and this workshop is just a vechicle to get you there. We hope that you find it to be a productive and enjoyable learning experience. In this workshop we will walk you through how to build a deep learning model to add fun emojis to sentences.

<span style="color:#FFFFFF">I love this workshop</span>

<span style="color:#FFFFFF">I love this workshop </span>  <span style="color:#000000"> __🥰__ </span>


# How to proceed?

# 

<span style="color:#000000">Fetching Our Data</span>

<span style="color:#000000">Preparing the Data</span>

<span style="color:#000000">Building the Model</span>

<span style="color:#000000">Train and Test</span>

# Fetching Data

# Find Good Data!!

<span style="color:#000000">We recommend Kaggle</span>

<span style="color:#000000">Lots of large and varied datasets</span>

<span style="color:#000000">Very easy to implement</span>

<img src="img/ok0.jpg" style="max-width:20px;" />

# 

# Choosing the Right Data

<span style="color:#000000">You need lots of data \(millions\)</span>

<span style="color:#000000">Watch out for unbalanced datasets</span>

<span style="color:#000000">Know what will be most useful </span>

<img src="img/ok1.jpg" style="max-width:20px;" />

# Data Processing

# Our Goals

<span style="color:#000000">Clean</span>

<span style="color:#000000">Classify</span>

<span style="color:#000000">Balance</span>

<span style="color:#000000">Make Machine Readable</span>

<img src="img/ok2.png" style="max-width:20px;" />

# 

# Cleaning

<span style="color:#000000">Normalise our Data \(e\.g lower case\)</span>

<span style="color:#000000">Keep the essential \(e\.g removing “stop words”\)</span>

<span style="color:#000000">NLTK package</span>

<img src="img/ok3.jpg" style="max-width:20px;" />

# 

# Classification

<span style="color:#000000">Emojis define our classes</span>

<span style="color:#000000">Separate emoji from text</span>

<img src="img/ok4.png" style="max-width:20px;" />

# 

# Balancing Dataset

<span style="color:#000000">Remove classes with insufficient text</span>

<span style="color:#000000">Imbalance leads to overfitting</span>

<span style="color:#000000">We want the same amount of data for every class </span>

# 

# Transforming Tweets

* <span style="color:#000000">Strings can’t be used\, so we must transform them</span>
* <span style="color:#000000">Vectorise the data : </span>
  * <span style="color:#000000">Make it part of our model</span>
  * <span style="color:#000000">Transform text dataset</span>
* <span style="color:#000000">We will apply the transformation on the dataset</span>

# 

# Transforming Tweets - Keras text tokenizer

* <span style="color:#000000">We use the Tokenizer class to create a word\-to\-index dictionary</span>
* <span style="color:#000000">Fit on texts </span>
  * <span style="color:#000000">Creates vocabulary index based on word frequency</span>
  * <span style="color:#000000">e\.g “The car drove around the track”</span>
    * <span style="color:#000000">dict\[“the”\] = 1</span>
    * <span style="color:#000000">dict\[“car”\] = 2</span>
* <span style="color:#000000">Texts to sequences</span>
  * <span style="color:#000000"> _Transforms each text into a sequence of integers_ </span>

# 

# Encoding Outputs

<span style="color:#000000">One Hot Encoding</span>

<span style="color:#000000">Easy and Effective</span>

<img src="img/ok5.png" style="max-width:20px;" />

# Building the Model

# What do we want from our model?

<span style="color:#000000">Handles textual data</span>

<span style="color:#000000">Takes on sequential data</span>

<span style="color:#000000">Considers context</span>

<span style="color:#000000">Efficient</span>

<span style="color:#000000">Doesn’t overcomplicate</span>

# Vectorisation

## Word Embeddings


<span style="color:#000000">Words are assigned real\-valued vectors</span>

<span style="color:#000000">They hold context and semantic</span>

<span style="color:#000000">Similar words are close together</span>

<img src="img/ok6.png" style="max-width:20px;" />


## One hot Encoding


<span style="color:#000000">Creating a zero vector with equal then place 1 in the index that corresponds to the word</span>

<span style="color:#000000">Inefficient : most elements are 0</span>

<img src="img/ok7.png" style="max-width:20px;" />

# 

# Word Embeddings

* <span style="color:#000000">Represents implicit relationships between words</span>
  * <span style="color:#000000">Helps us gain contextual information</span>
  * <span style="color:#000000">Similar words have similar encoding</span>
  * <span style="color:#000000">Boosts generalisation and performance</span>
* <span style="color:#000000">Pre\-trained model : GloVe \(Global Vectors for Word Representation\)</span>

# 

# Embedding Layer

* <span style="color:#000000">Create a lookup table which relates a word to its corresponding word embedding</span>
* <span style="color:#000000">Attach matrix to the keras Embedding Layer</span>
* <span style="color:#000000">Make it easier to take on large inputs</span>
  * <span style="color:#000000">e\.g sparse vectors representing words</span>
* <span style="color:#000000">Captures the semantic of sentence effectively</span>

# 

# Spatial Dropout 1D

<img src="img/ok8.png" style="max-width:20px;" />

<span style="color:#000000">Prevent overfitting</span>

<span style="color:#000000">Spatial Dropout drops entire feature maps</span>

<span style="color:#000000">Makes our model dropout entire phrases and forces it to generalise better</span>

# 

# Convolutional Layer

* <span style="color:#000000">Performs window\-based feature extraction</span>
  * <span style="color:#000000">e\.g patterns in sequential word groupings indicating certain emotions</span>
* <span style="color:#000000">Trains to pull out the essential of a sentence \(attention\)</span>

<img src="img/ok9.gif" style="max-width:20px;" />

# 

# Bidirectional LSTM

* <span style="color:#000000">Takes current and preceding inputs into consideration</span>
* <span style="color:#000000">Able to catch long term dependencies</span>
* <span style="color:#000000">Reads sentence in both directions</span>
  * <span style="color:#000000">Incorporating future time steps helps understand context better</span>

# 

# Output Layer

* <span style="color:#000000">Softmax activation function</span>
  * <span style="color:#000000">best for multi\-class classification</span>
* <span style="color:#000000">Returns one hot array representing emoji</span>

