
# Emojification
I made this notebook for when I was giving a workshop in collaboration with Google Cloud at the AI+X Summit 2022 in Zurich. This was meant to introduce NLP problems in Deep Learning to the audience. The example we used was a bot that recommends emojis based on the user's input sentence. 
# Introduction
The ACE workshop is built around the idea that the best way to learn something new is by getting a hands-on experience. Leanring something new is a journey form where you are now to where you want to be and this workshop is just a vechicle to get you there. We hope that you find it to be a productive and enjoyable learning experience. In this workshop we will walk you through how to build a deep learning model to add fun emojis to sentences.

We need to train our model on sentences with emojis so that our model learns when to use what emoji. For this task we will be using Kaggle to collect the data. There are multiple options the best ones in my opinion are the datasets which are collections of english language tweets. So either you can filter all tweets to keep only the ones containing at least 1 emoji or we can just use an already filtered dataset [EmojifyData-EN: English tweets, with emojis](https://www.kaggle.com/datasets/rexhaif/emojifydata-en).

* * *

After having created an account which you can do by clicking [here](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) or you can sign in by clicking [here](https://www.kaggle.com/account/login?phase=startSignInTab&returnUrl=%2F), you must download Kaggle's beta API which you can do by going to your account settings and clicking on "Create a New API Token". This will download a file called "kaggle.json" to your computer.

We must now make sure that we mount our Google Drive files by running the following code:


Mounted at /content/gdrive

Now we can upload the "kaggle.json" file from our computer to our notebook.

We will be now downloading the tweets from kaggle to our notebook and unzip the folder.



What does our data tell us?[¶](#What-does-our-data-tell-us?)
------------------------------------------------------------

We know what the data approxiamtely looks like thanks its website on Kaggle. Every new tweet starts with Start tag and at the end there's STOP tag. Every word is seperated by "0\\n". Furthermore, we see that the emojis of the tweets are in CLDR Short Name format e.g a red heart is denoted as :red\_heart:

Our Goal[¶](#Our-Goal)
----------------------

We want to create a dataframe which will have 2 columns. One denoting a tweet without its original emoji and the other one containing the corresponding emoji. So we must seperate all emojis from their corresponding tweets e.g. we want to start from

> Congratulations Mo has been named the Players Player of the Year :clapping\_hands:

and end with

> Congratulations Mo has been named the Players Player of the Year
> 
> :clapping\_hands:

Step by Step[¶](#Step-by-Step)
------------------------------

### 1) Remove all unecessary tags[¶](#1)-Remove-all-unecessary-tags)


### 2) Seperate text into seperate tweets and place them into dataframe[¶](#2)-Seperate-text-into-seperate-tweets-and-place-them-into-dataframe)



|   |tweets                                           |
|---|-------------------------------------------------|
|0  |No object is so beautiful that under certain c...|
|1  |Cant expect different results doing the same t...|
|2  |“ Lets go Marcus ” “ Shiiit where we goin Home...|
|3  |Asahd really is a grown man in the body of a 1...|
|4  |Yoongi Tweet Hello Im Min fell on Butt What th...|


### 3) Seperate the emojis from the tweets[¶](#3)-Seperate-the-emojis-from-the-tweets)


|   |text                                             |emoji                 |
|---|-------------------------------------------------|----------------------|
|0  |No object is so beautiful that under certain c...|red_heart             |
|1  |Cant expect different results doing the same t...|person_shrugging      |
|2  |“ Lets go Marcus ” “ Shiiit where we goin Home...|face_with_tears_of_joy|
|3  |Asahd really is a grown man in the body of a 1...|face_with_tears_of_joy|
|4  |Yoongi Tweet Hello Im Min fell on Butt What th...|face_with_tears_of_joy|


### 4) Delete all unwanted Emojis from Dataframe[¶](#4)-Delete-all-unwanted-Emojis-from-Dataframe)


### 5) Balance and Shuffle the Data[¶](#5)-Balance-and-Shuffle-the-Data)




|   |text                                             |emoji                         |
|---|-------------------------------------------------|------------------------------|
|0  |i like this one Keep it up                       |clapping_hands                |
|1  |My Little Brother He Is nly 12 Years Doing Gra...|loudly_crying_face            |
|2  |You ready to drop out and start a skate shop i...|smiling_face_with_sunglasses  |
|3  |happy birthday my dude hope youve had a great ...|smiling_face_with_smiling_eyes|
|4  |Lester Holt amp NBC BIG BB                       |flushed_face                  |


### 6) Clean the tweets homogenuously but don't change the semantics[¶](#6)-Clean-the-tweets-homogenuously-but-don't-change-the-semantics)

|   |text                                             |emoji                         |
|---|-------------------------------------------------|------------------------------|
|0  |like one keep                                    |clapping_hands                |
|1  |little brother only 12 years grade 6 fighting ...|loudly_crying_face            |
|2  |ready drop sta skate shop queens                 |smiling_face_with_sunglasses  |
|3  |happy bihday dude hope youve great day today     |smiling_face_with_smiling_eyes|
|4  |letter holt amp nbc big bb                       |flushed_face                  |


At this point we've cleaned and organised our data. Now we have to transform our data such that our neural network will be able to use it to train.

Transforming Tweets[¶](#Transforming-Tweets)
--------------------------------------------

Our first challenge is to transform the tweets into a readable form for our model. We will be using the keras text tokenizer which allows to vectorize a text corpus, by turning each text into a sequence of integers (each integer being the index of a token in a dictionary), we will then pad these sequences so that all our input data is uniform. Finally we will transform the emojis into one hot sequences.



The first layer of our model will be an embedding one. An embedding layer enables us to convert each word into a fixed length vector of defined size. The resultant vector is a dense one having real values instead of just 0's and 1's. The fixed length of word vectors helps us to represent words in a better way along with reduced dimensions. An LSTM model generally works well for such a text classification problem. However, it takes forever to train. One way to speed up the training time is to improve the network adding “Convolutional” layer. Convolutional Neural Networks (CNN) come from image processing. They pass a “filter” over the data and calculate a higher-level representation. They have been shown to work surprisingly well for text, even though they have none of the sequence processing ability of LSTMs. Moreover, to increase the value of the data we're going to turn our LSTM layer to a bidirectional one. This is done so that a cell can be used to train two sides, instead of one side of the input sequence.This provides one more context to the word to fit in the right context from words coming after and before, resulting in faster and fully learning and solving a problem.

Building the Embedding Layer[¶](#Building-the-Embedding-Layer)
--------------------------------------------------------------

TensorFlow enables you to train word embeddings. However, this process not only requires a lot of data but can also be time and resource-intensive. To tackle these challenges you can use pre-trained word embeddings. Let's illustrate how to do this using GloVe (Global Vectors) word embeddings by Stanford. These embeddings are obtained from representing words that are similar in the same vector space. This is to say that words that are negative would be clustered close to each other and so will positive ones.

The first step is to obtain the word embedding and append them to a dictionary. After that, you'll need to create an embedding matrix for each word in the training set. Let's start by downloading the GloVe word embeddings.


The next step is to create a word embedding matrix for each word in the word index that you obtained earlier. If a word doesn't have an embedding in GloVe it will be presented with a zero matrix.


The next step is to use the embedding you obtained above as the weights to a Keras embedding layer. You also have to set the trainable parameter of this layer to False so that is not trained. There are a couple of other things to note:

*   The Embedding layer takes the first argument as the size of the vocabulary. 1 is added because 0 is usually reserved for padding
*   The input\_length is the length of the input sequences
*   The output\_dim is the dimension of the dense embedding



We have a trained model now. But to diversiy our outputs a bit more instead of selecting te emoji that our model assigned the greatest confidence, we will pick a random emoji from the weighted confidence output array. So now our final decision is not only determined by our model but there is also a certain element of chance that comes to it so for example the emoji with the second highest confidence is chosen. We will be using the emoji package which displays the emojis for us.
