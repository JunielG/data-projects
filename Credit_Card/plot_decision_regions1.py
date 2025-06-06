def plot_decision_regions(X, y, clf, ax=None, resolution=0.02, title='', markers='soD^v<>', colors='rymbgc'):
    """
    Plot decision regions of a classifier.
    
    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Feature matrix of the dataset.
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
    markers : str (default: 'soD^v<>')
        Markers to use for different classes.
    colors : str (default: 'rymbgc')
        Colors to use for different classes.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the decision regions.
    """
    
    from matplotlib.colors import ListedColormap
    # Check if X has only two features
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
    
    # Define custom colormap for the regions
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#FFAAFF', '#AAFFFF'][:n_classes])
    
    # Define markers and colors for scatter plot
    class_markers = list(markers[:n_classes])
    class_colors = list(colors[:n_classes])
    
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
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    # Plot class samples
    for idx, cl in enumerate(unique_classes):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1], 
                   alpha=0.8, c=class_colors[idx], 
                   marker=class_markers[idx], label=f'Class {cl}')
    
    # Add legend
    ax.legend(loc='best')
    
    return ax