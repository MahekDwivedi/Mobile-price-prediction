import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline



# dataset from kaggle
df=pd.read_csv('../input/mobile-price-classification/train.csv')
df.head()



from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow');



corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim()



df.describe()

#finding any null values
df.isnull().any()

sns.pairplot(df)


#analyzing data using plots 

plt.figure(figsize=(12,4))
sns.heatmap(df.corr())    #correlation heatmap is a heatmap that shows a 2D correlation matrix between two discrete dimensions
plt.show()


# price classification on the basis oftouch screen
plt.figure(figsize=(12,4))
sns.barplot(x='price_range' ,y='ram',hue='touch_screen', data=df, palette="Reds")
plt.show()


plt.figure(figsize=(12,4))
sns.lineplot(data=df, x='price_range', y='four_g' ,hue='dual_sim', palette='ocean')
plt.show()


