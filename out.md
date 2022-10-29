![](img/ok0.png)

![](img/ok1.gif)

![](img/ok2.gif)

<span style="color:#000000"> __Vertex AI__ </span>

<span style="color:#4285F4">Put the best of Google’s AI technologies to work</span>

<span style="color:#202124">Unified development and deployment platform for data science and machine learning</span>

<span style="color:#202124">Oct 2022</span>

![](img/ok3.png)

![](img/ok4.png)

![](img/ok5.png)

![](img/ok6.png)

_Vertex AI Pre\-Trained Models_

<span style="color:#272A2C">Generally available</span>

![](img/ok7.png)

![](img/ok8.png)

![](img/ok9.png)

    * <span style="color:#272A2C">AutoML Translation</span>

![](img/ok10.png)

![](img/ok11.png)

![](img/ok12.png)

    * <span style="color:#272A2C">Video Intelligence</span>

    * <span style="color:#272A2C">Natural Language</span>

![](img/ok13.png)

![](img/ok14.png)

    * <span style="color:#272A2C">AutoML</span>  <span style="color:#272A2C">Video Intelligence</span>

![](img/ok15.png)

    * <span style="color:#272A2C">AutoML</span>  <span style="color:#272A2C">Natural Language</span>

<span style="color:#272A2C">Best in class tools allowing customers to leverage Google’s leadership in AI to solve common problems </span>

<span style="color:#272A2C">Structured data</span>

    * <span style="color:#272A2C">Time Series Insights API</span>

    * <span style="color:#272A2C">Vertex AI Forecast</span>

    * <span style="color:#272A2C">Fleet Routing API</span>

![](img/ok16.png)

<span style="color:#272A2C">No code / low code workflow</span>

<span style="color:#272A2C">Data Science tool kit</span>

<span style="color:#FFFFFF"> __Integration with__ </span>

<span style="color:#FFFFFF"> __Data Services__ </span>

<span style="color:#272A2C">Unified development and deployment platform for data science and machine learning</span>

<span style="color:#272A2C">Increase productivity of data scientists and ML engineers</span>

<span style="color:#FFFFFF">Model Monitoring</span>

<span style="color:#4285F4"> __Vertex AI Workbench__ </span>

<span style="color:#3C4043">A one\-stop surface for Data Science</span>

![](img/ok17.png)

<span style="color:#202124"> __Fully managed compute with admin control__ </span>

<span style="color:#202124">A Jupyter\-based fully managed\, scalable\, enterprise\-ready compute infrastructure with easily enforceable policies and user management</span>

<span style="color:#202124"> __Fast workflow for data tasks__ </span>

<span style="color:#202124">Seamless visual and code\-based integrations with data & analytics services</span>

![](img/ok18.png)

<span style="color:#202124"> __At\-your\-fingertips integration__ </span>  <span style="color:#202124"> </span>

<span style="color:#202124">Load and share notebooks alongside your AI and data tasks\. Run tasks without extra code</span>

# Vertex AI Set Up Instruction

Task 1: Create a notebook instance in Vertex AI Workbench

1\.  <span style="color:#202124">From the Google Cloud Console </span>  <span style="color:#202124"> __Navigation Menu__ </span>  <span style="color:#202124">\, select </span>  <span style="color:#202124"> __Vertex AI__ </span>  <span style="color:#202124">\. From the Dashboard\, select the </span>  <span style="color:#202124"> __Workbench__ </span>  <span style="color:#202124"> menu item</span>

2\. In the  __User\-Managed Notebooks__  tab\,  <span style="color:#202124">press the </span>  <span style="color:#202124"> __\+ New Instance__ </span>  <span style="color:#202124"> button at the top of the screen\.</span>

3\.  <span style="color:#202124">Select </span>  <span style="color:#202124"> __TensorFlow Enterprise __ </span>  <span style="color:#202124">></span>  <span style="color:#202124"> __ TensorFlow Enterprise 2\.6 \(with LTS\) __ </span>  <span style="color:#202124">></span>  <span style="color:#202124"> __ Without GPUs__ </span>  <span style="color:#202124">:</span>

4\.  <span style="color:#202124">In the pop\-up\, confirm the name of the deep learning VM\, move to the bottom of the window and click </span>  <span style="color:#202124"> __CREATE__ </span>  <span style="color:#202124">:</span>

![](img/ok19.png)

![](img/ok20.png)

![](img/ok21.png)

Task 1: Create a notebook instance in Vertex AI Workbench

5\.  <span style="color:#202124">The notebook instance will now be provisioned\. </span>

<span style="color:#202124">This typically takes 3\-5 minutes\.</span>

6\. When it is ready click the   __OPEN JUPYTERLAB__  link\.

![](img/ok22.png)

![](img/ok23.png)

Task 2: Copy the notebook into your Vertex Workbench instance

<span style="color:#202124">1\. In JupyterLab\, click the Terminal icon to open a new terminal\.</span>

<span style="color:#202124">2\. At the command\-line prompt\, type in the following command and press Enter\.</span>

<span style="color:#212121">Use the </span>  <span style="color:#37474F">gsutil cp</span>  <span style="color:#212121"> command to upload the training notebook as well as an image that will be used to your notebook instance:</span>

<span style="color:#202124">3\. Confirm that you have copied both items by ensuring that you can see it in your directory on the left\.</span>

| gsutil cp gs://ta-reinforecement-learning/aix/Emojification.ipynb . |
| :-: |


| gsutil cp gs://ta-reinforecement-learning/aix/Emojification.ipynb . |
| :-: |


# Additional Lab to Practice NLP and Vertex AI (by D-Labs)

![](img/ok24.png)

__Sentiment Analysis using BERT on Vertex AI __ \(incl\. TFX and Vertex Pipelines\)

_[https://dlabs\.ai/resources/courses/bert\-sentiment\-analysis\-on\-vertex\-ai\-using\-tfx/](https://dlabs.ai/resources/courses/bert-sentiment-analysis-on-vertex-ai-using-tfx/)_

# Tackle Any NLP Problem

# Adding Emojis to your Sentences

I love this workshop  __🥰__

I love this workshop

# How to proceed?

# 

Fetching Our Data

Preparing the Data

Building the Model

Train and Test

# Fetching Data

# Find Good Data!!

We recommend Kaggle

Lots of large and varied datasets

Very easy to implement

![](img/ok25.png)

![](img/ok26.jpg)

# Fetching Data

# Choosing the Right Data

You need lots of data \(millions\)

Watch out for unbalanced datasets

Know what will be most useful

![](img/ok27.jpg)

![](img/ok28.png)

# Data Processing

# Our Goals

Clean

Classify

Balance

Make Machine Readable

![](img/ok29.png)

# Data Processing

# Cleaning

Normalise our Data \(e\.g lower case\)

Keep the essential \(e\.g removing “stop words”\)

NLTK package

![](img/ok30.jpg)

# Data Processing

# Classification

Emojis define our classes

Separate emoji from text

![](img/ok31.png)

# Data Processing

# Balancing Dataset

Remove classes with insufficient text

Imbalance leads to overfitting

We want the same amount of data for every class

![](img/ok32.jpg)

# Data Processing

# Transforming Tweets

* Strings can’t be used\, so we must transform them
* Vectorise the data :
  * Make it part of our model
  * Transform text dataset
* We will apply the transformation on the dataset

# Data Processing

# Transforming Tweets - Keras text tokenizer

* We use the Tokenizer class to create a word\-to\-index dictionary
* Fit on texts
  * Creates vocabulary index based on word frequency
  * e\.g “The car drove around the track”
    * dict\[“the”\] = 1
    * dict\[“car”\] = 2
* Texts to sequences
  * _Transforms each text into a sequence of integers_

# Data Processing

# Encoding Outputs

One Hot Encoding

Easy and Effective

![](img/ok33.png)

# Building the Model

# What do we want from our model?

Handles textual data

Takes on sequential data

Considers context

Efficient

Doesn’t overcomplicate

# Building the Model - Vectorisation

# Word Embeddings

# One hot Encoding

Words are assigned real\-valued vectors

They hold context and semantic

Similar words are close together

Creating a zero vector with equal then place 1 in the index that corresponds to the word

Inefficient : most elements are 0

![](img/ok34.png)

![](img/ok35.png)

# Building the Model

# Word Embeddings

* Represents implicit relationships between words
  * Helps us gain contextual information
  * Similar words have similar encoding
  * Boosts generalisation and performance
* Pre\-trained model : GloVe \(Global Vectors for Word Representation\)

# Building the Model

# Embedding Layer

* Create a lookup table which relates a word to its corresponding word embedding
* Attach matrix to the keras Embedding Layer
* Make it easier to take on large inputs
  * e\.g sparse vectors representing words
* Captures the semantic of sentence effectively

# Building the Model

# Spatial Dropout 1D

![](img/ok36.png)

Prevent overfitting

Spatial Dropout drops entire feature maps

Makes our model dropout entire phrases and forces it to generalise better

# Building the Model

![](img/ok37.png)

# Convolutional Layer

* Performs window\-based feature extraction
  * e\.g patterns in sequential word groupings indicating certain emotions
* Trains to pull out the essential of a sentence \(attention\)

![](img/ok38.gif)

# Building the Model

# Bidirectional LSTM

![](img/ok39.png)

* Takes current and preceding inputs into consideration
* Able to catch long term dependencies
* Reads sentence in both directions
  * Incorporating future time steps helps understand context better

# Building the Model

# Output Layer

![](img/ok40.png)

* Softmax activation function
  * best for multi\-class classification
* Returns one hot array representing emoji

# How to Upload a Notebook

# Vertex AI

# Google Colab

_[https://blog\.tensorflow\.org/2022/05/5\-steps\-to\-go\-from\-notebook\-to\-deployed\.html](https://blog.tensorflow.org/2022/05/5-steps-to-go-from-notebook-to-deployed.html)_

_[https://stackoverflow\.com/questions/48849938/how\-to\-directly\-upload\-a\-jupyter\-notebook\-from\-local\-machine\-onto\-google\-collab](https://stackoverflow.com/questions/48849938/how-to-directly-upload-a-jupyter-notebook-from-local-machine-onto-google-collab)_

