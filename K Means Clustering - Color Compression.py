# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:11:51 2020

@author: Ishan
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
X,y_true=make_blobs(n_samples=300,centers=4,cluster_std=0.60,random_state=0)
print(y_true)
plt.scatter(X[:,0],X[:,1],s=50)

kmeans=KMeans(n_clusters=4)
kmeans.fit(X)
y_means=kmeans.predict(X)
print(y_means)
def find_clusters(X,n_clusters,rseed=2):
    rng=np.random.RandomState(rseed)
    i=rng.permutation(X.shape[0])[:n_clusters]
    centers=X[i]
    while True:
        labels=pairwise_distances_argmin(X,centers)
        new_centers=np.array([X[labels==i].mean(0)
                             for i in range(n_clusters)])
    
        if np.all(centers==new_centers):
            break
        centers=new_centers
    return centers,labels
centers,labels=find_clusters(X,4)
plt.scatter(X[:,0],X[:,1],c=y_means,s=50,cmap="viridis")
plt.scatter(centers[:,0],centers[:,1],c="black",s=200,alpha=0.5)

from sklearn.datasets import load_sample_image
china=load_sample_image("flower.jpg")
ax=plt.axes(xticks=[],yticks=[])
ax.imshow(china)

data=china/255.0
data=data.reshape(427*640,3)
data.shape

def plot_pixels(data,title,colors=None,N=10000):
    title=title
    if colors is None:
        colors=data
    rng=np.random.RandomState(0)
    i=rng.permutation(data.shape[0])[:N]
    colors=colors[i]
    r,g,b=data[i].T
    fig,ax=plt.subplots(1,2,figsize=(16,6))
    
    ax[0].scatter(r,g,color=colors,marker='.')
    ax[0].set(xlabel="Red",ylabel="green",xlim=(0,1),ylim=(0,1))
    
    ax[1].scatter(r,b,color=colors,marker='.')
    ax[1].set(xlabel="Red",ylabel="Blue",xlim=(0,1),ylim=(0,1))
    fig.suptitle(title,size=10)
plot_pixels(data,title="Input color space is 16 million")

import warnings;warnings.simplefilter("ignore")

from sklearn.cluster import MiniBatchKMeans
kmeans=MiniBatchKMeans(16)
kmeans.fit(data)
new_colors=kmeans.cluster_centers_[kmeans.predict(data)]
#plot_pixels(data,colors=new_colors,title="Reduced to 16 color space")
china_recolored=new_colors.reshape(china.shape)
fig,ax=plt.subplots(1,2,figsize=(16,6),subplot_kw=dict(xticks=[],yticks=[0]))
fig.subplots_adjust(wspace=0.5)
ax[0].imshow(china)
ax[1].imshow(china_recolored)




from sklearn.datasets import load_sample_image
china=load_sample_image("china.jpg")
ax=plt.axes(xticks=[],yticks=[])
ax.imshow(china)

data=china/255.0
data=data.reshape(427*640,3)
data.shape

def plot_pixels(data,title,colors=None,N=10000):
    title=title
    if colors is None:
        colors=data
    rng=np.random.RandomState(0)
    i=rng.permutation(data.shape[0])[:N]
    colors=colors[i]
    r,g,b=data[i].T
    fig,ax=plt.subplots(1,2,figsize=(16,6))
    
    ax[0].scatter(r,g,color=colors,marker='.')
    ax[0].set(xlabel="Red",ylabel="green",xlim=(0,1),ylim=(0,1))
    
    ax[1].scatter(r,b,color=colors,marker='.')
    ax[1].set(xlabel="Red",ylabel="Blue",xlim=(0,1),ylim=(0,1))
    fig.suptitle(title,size=10)
plot_pixels(data,title="Input color space is 16 million")

import warnings;warnings.simplefilter("ignore")

from sklearn.cluster import MiniBatchKMeans
kmeans=MiniBatchKMeans(16)
kmeans.fit(data)
new_colors=kmeans.cluster_centers_[kmeans.predict(data)]
#plot_pixels(data,colors=new_colors,title="Reduced to 16 color space")
china_recolored=new_colors.reshape(china.shape)
fig,ax=plt.subplots(1,2,figsize=(16,6),subplot_kw=dict(xticks=[],yticks=[0]))
fig.subplots_adjust(wspace=0.5)
ax[0].imshow(china)
ax[1].imshow(china_recolored)















