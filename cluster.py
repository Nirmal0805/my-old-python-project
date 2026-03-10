#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load in our packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

from sklearn.cluster import KMeans

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Connect to our case study data set

myClusterData = pd.read_csv("D:\\Machine learning\\Data\\cluster-py.csv")


# In[3]:


# Have a look at our data
myClusterData.head(20)


# In[4]:


# Plot our b1 & b3 data
plt.scatter(myClusterData.b1,myClusterData.b3)


# In[5]:


# Assign values to x & y x as b3 and y as b1

x = [16, 25, 18, 22, 5, 10, 21, 4, 30, 25]
y = [11, 7, 9, 16, 16, 15, 16, 7, 17, 5]


# In[6]:


# Plot our x & y values
plt.scatter(x,y)
plt.show()


# In[7]:


test = np.array([myClusterData.b1,myClusterData.b3])
test


# In[8]:


plt.scatter(test[[0]],test[[1]])
plt.show()


# In[17]:



# Pivot our data to work as an array 
## Sampled some data from b1 and b3 columns.
X = np.array([[16, 11],
              [25, 7],
              [18, 9],
              [22, 16],
              [5, 16],
              [10, 15],
              [21, 16],
              [4, 7],
              [30, 17],
              [25, 5]])


# In[15]:


# Assign the value of n clusters / run the algorithm / assign centroids / label our group names 
from sklearn.cluster import KMeans

mygroups = KMeans(n_clusters=3)
mygroups


# In[18]:


mygroups.fit(X)


# In[19]:


centriods = mygroups.cluster_centers_
labels = mygroups.labels_


# In[21]:


# Set up our color palette

colors = ["b.","g.","r.","c.","m."]

# Plot each point

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

# Generate the view

plt.scatter(centriods[:, 0],centriods[:, 1], marker = "x", s=150, linewidths = 5)
plt.show()


# In[ ]:




