def plot_decision_regions(X, y, clf, ax=None, resolution=0.02, title=''):
    """
    Plot decision regions of a classifier in the style of the reference image.
    
    Parameters
    ----------
    X : array-like, shape = [n_samples, 2]
        Feature matrix of the dataset (must have exactly 2 features).
    y : array-like, shape = [n_samples]
        Target values.
    clf : classifier object
        Must have a .predict method.
    ax : matplotlib.axes.Axes (default: None)
        The axes where the decision regions will be plotted. If None, a new figure
        and axes will be created.
    resolution : float (default: 0.02)
        Grid resolution for the decision boundaries.
    title : str (default: '')
        Plot title.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the decision regions.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    # Check if X has exactly two features
    if X.shape[1] != 2:
        raise ValueError('X must have exactly two features for visualization.')
    
    # Create new axes if needed
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Set title if provided
    if title:
        ax.set_title(title)
    
    # Get unique class labels
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    # Define colors matching the reference image
    # Light blue, light orange, light green color scheme
    colors = ['#8FBBD9', '#F9BC8F', '#AEDCAB']
    cmap = ListedColormap(colors[:n_classes])
    
    # Define markers and edge colors for scatter plot matching the reference image
    markers = ['s', '^', 'o']  # square, triangle, circle
    edge_colors = ['#4878A4', '#A65628', '#5CA255']  # darker versions of the fill colors
    
    # Get min and max values for the two features
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # Predict class labels for each point in the mesh grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision regions
    ax.contourf(xx, yy, Z, alpha=1.0, cmap=cmap)
    
    # Set axis limits
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    # Plot class samples
    for idx, cl in enumerate(unique_classes):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1], 
                  alpha=1.0, 
                  c=edge_colors[idx],
                  marker=markers[idx], 
                  edgecolors=edge_colors[idx],
                  s=40,
                  linewidth=1.5,
                  label=f'{cl}')
    
    # Add a legend to top-right corner
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', frameon=False)
    
    # Remove axis ticks and labels for clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    return ax


clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)

classifiers = [clf1, clf2, clf3]
titles = ['Logistic Regression', 'Random Forest', 'Naive Bayes']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(wspace=0.1)

for clf, title, ax in zip(classifiers, titles, axes.flatten()):
    plot_decision_regions(X, y, clf=clf, ax=ax, title=title)
    
plt.tight_layout()
plt.savefig('3_classifiers.png', dpi=300, bbox_inches='tight')
plt.show()