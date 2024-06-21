# https://www.geeksforgeeks.org/music-recommendation-system-using-machine-learning/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.manifold import TSNE

import warnings

warnings.filterwarnings('ignore')

tracks = pd.read_csv('tracks.csv')

# tracks.head()

# tracks.shape

tracks.info()

tracks.isnull().sum()

model = TSNE(n_components= 2, random_state=0, perplexity=4, learning_rate='auto')
X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
tsne_data = model.fit_transform(X)
plt.figure(figsize = (8,8))
plt.scatter(tsne_data[:,0], tsne_data[:,1])
plt.show()

