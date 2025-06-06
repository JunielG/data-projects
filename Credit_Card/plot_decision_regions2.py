def plot_decision_regions(X, y, clf, ax=None, resolution=0.02, title='', markers='soD^v<>', colors='rymbgc', feature_indices=None, apply_pca=True):
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
    feature_indices : tuple of int (default: None)
        Indices of the two features to use for visualization. If None and apply_pca is False,
        the first two features will be used.
    apply_pca : bool (default: True)
        Whether to apply PCA to reduce the data to 2D if it has more than 2 features.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the decision regions.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.decomposition import PCA
    
    # Handle case where X has more than two features
    X_visual = X.copy()
    
    if X.shape[1] > 2:
        if apply_pca:
            # Apply PCA for visualization
            pca = PCA(n_components=2)
            X_visual = pca.fit_transform(X)
            feature_names = ['PCA Component 1', 'PCA Component 2']
        elif feature_indices is not None:
            # Use specified features
            X_visual = X[:, feature_indices]
            feature_names = [f'Feature {feature_indices[0]}', f'Feature {feature_indices[1]}']
        else:
            # Default to first two features
            X_visual = X[:, :2]
            feature_names = ['Feature 0', 'Feature 1']
    else:
        feature_names = ['Feature 0', 'Feature 1']
    
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
    x_min, x_max = X_visual[:, 0].min() - 1, X_visual[:, 0].max() + 1
    y_min, y_max = X_visual[:, 1].min() - 1, X_visual[:, 1].max() + 1
    
    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # Prepare the mesh grid points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # For high-dimensional data, we need to transform mesh points back to original space
    if X.shape[1] > 2 and apply_pca:
        # This is an approximation, as we can't perfectly map from PCA space back to original space
        # We'll use the pseudo-inverse of the PCA components
        mesh_orig_shape = pca.components_.shape
        # Using Moore-Penrose pseudo-inverse
        pinv = np.linalg.pinv(pca.components_)
        # Transform mesh points back to original space
        mesh_points_orig_space = np.dot(mesh_points - pca.mean_, pinv)
        Z = clf.predict(mesh_points_orig_space)
    else:
        # If using original features or not applying PCA, predict directly
        if X.shape[1] == 2 or not apply_pca:
            Z = clf.predict(mesh_points)
        else:
            # If using just 2 selected features from a higher dim dataset,
            # we need to create full feature vectors with default values
            default_values = np.tile(np.mean(X, axis=0), (mesh_points.shape[0], 1))
            if feature_indices:
                default_values[:, feature_indices[0]] = mesh_points[:, 0]
                default_values[:, feature_indices[1]] = mesh_points[:, 1]
            else:
                default_values[:, 0] = mesh_points[:, 0]
                default_values[:, 1] = mesh_points[:, 1]
            Z = clf.predict(default_values)
    
    Z = Z.reshape(xx.shape)
    
    # Plot decision regions
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    # Plot class samples
    for idx, cl in enumerate(unique_classes):
        ax.scatter(x=X_visual[y == cl, 0], y=X_visual[y == cl, 1], 
                   alpha=0.8, c=class_colors[idx], 
                   marker=class_markers[idx], label=f'Class {cl}')
    
    # Add legend and axis labels
    ax.legend(loc='best')
    if not ax.get_xlabel():
        ax.set_xlabel(feature_names[0])
    if not ax.get_ylabel():
        ax.set_ylabel(feature_names[1])
    
    return ax