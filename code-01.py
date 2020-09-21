import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
# %matplotlib inline

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
RS = 123

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.mnist_reader import load_mnist

X_train, y_train = load_mnist('./data/fashion', kind='train')
print('X_train.shape ->', X_train.shape)
print('y_train ->\n', y_train)

# Utility function to visualize the outputs of PCA and t-SNE
def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

# Subset first 20k data points to visualize
x_subset = X_train[0:20000]
y_subset = y_train[0:20000]
print('np.unique(y_subset) ->', np.unique(y_subset))

# PCA Visualization
time_start = time.time()
pca = PCA(n_components=4)
pca_result = pca.fit_transform(x_subset)
print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))
pca_df = pd.DataFrame(columns = ['pca1','pca2','pca3','pca4'])
pca_df['pca1'] = pca_result[:,0]
pca_df['pca2'] = pca_result[:,1]
pca_df['pca3'] = pca_result[:,2]
pca_df['pca4'] = pca_result[:,3]
print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))
top_two_comp = pca_df[['pca1','pca2']] # taking first and second principal component
print('# Visualizing the PCA output')
f, ax, sc, txts = fashion_scatter(top_two_comp.values,y_subset)
print(f, ax, sc, txts, sep='\n') # Visualizing the PCA output

# t-SNE Visualization
time_start = time.time()
fashion_tsne = TSNE(random_state=RS).fit_transform(x_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
print('# Visualizing the t-SNE output')
fashion_scatter(fashion_tsne, y_subset)

# Recommended Approach, first PCA then t-SNE
time_start = time.time()
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(x_subset)
print('PCA with 50 components done! Time elapsed: {} seconds'.format(time.time()-time_start))
print('Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
time_start = time.time()
fashion_pca_tsne = TSNE(random_state=RS).fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
print('# Visualizing the PCA then t-SNE output')
fashion_scatter(fashion_pca_tsne, y_subset)
plt.show()






