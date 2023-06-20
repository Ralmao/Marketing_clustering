Marketing_clustering
Project Master Class Marketing Clustering AI with a neural networks and clean dataset like a pro
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import cv2
from skimage import io
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, normalize
import os
import glob
import random
from google.colab import files #Librería para cargar ficheros directamente en Colab
%matplotlib inline

Explaining the algorithm :

Clustering is an unsupervised machine learning algorithm that can be used to group similar data points together. Clustering algorithms do not have any target variables, and they are used to discover the underlying structure of the data.

There are many different clustering algorithms available, but some of the most popular ones include:

K-means clustering: This algorithm groups data points into k clusters, where k is a user-defined parameter. The algorithm works by iteratively assigning data points to the cluster with the nearest mean.
Hierarchical clustering: This algorithm builds a hierarchy of clusters by repeatedly merging or splitting clusters. The algorithm can be either agglomerative (merging clusters) or divisive (splitting clusters).
DBSCAN: This algorithm clusters data points that are densely connected together. The algorithm works by identifying clusters of points that are more densely connected than other points in the dataset.
Scikit-learn provides implementations of many different clustering algorithms. To use a clustering algorithm in scikit-learn, you first need to create an instance of the clustering algorithm class. Then, you need to fit the algorithm to the data. Once the algorithm is fitted, you can use it to predict the cluster labels for new data points.

Explaining the clustering algorithm with convolutional neural networks (CNNs) :

Clustering algorithms with convolutional neural networks (CNNs) are a type of unsupervised learning algorithm that can be used to group similar data points together. CNNs are a type of deep learning algorithm that are specifically designed for processing data that has a spatial or temporal structure. This makes them well-suited for clustering tasks, such as image clustering and text clustering.

There are two main ways to use CNNs for clustering tasks:

Feature extraction: In this approach, the CNN is used to extract features from the data. These features are then used to train a traditional clustering algorithm, such as K-means clustering.
Self-supervised learning: In this approach, the CNN is trained to learn a representation of the data that captures the underlying structure of the data. This representation can then be used to cluster the data points.
Feature extraction is a relatively straightforward approach to clustering with CNNs. However, it can be difficult to choose the right features to extract. Self-supervised learning is a more recent approach to clustering with CNNs. It is more challenging to implement, but it can be more effective than feature extraction.

Here are some of the benefits of using CNNs for clustering tasks:

CNNs are able to learn hierarchical representations of the data. This means that they can cluster data points that are not only similar in terms of their features, but also in terms of their relationships to other data points.
CNNs are able to handle large datasets. This is because they are able to learn features from the data in a hierarchical manner.
Here are some of the drawbacks of using CNNs for clustering tasks:

CNNs can be computationally expensive to train.
CNNs can be difficult to interpret.
Overall, CNNs are a powerful tool for clustering tasks. They are able to learn hierarchical representations of the data, and they can handle large datasets. However, they can be computationally expensive to train, and they can be difficult to interpret.
