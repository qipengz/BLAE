import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource, Normalize
import plotly.graph_objects as go
from plotly.offline import iplot
from torchvision.utils import make_grid



def plot_interactive_3d(model, data, labels, cmap='viridis', figsize=(10, 8), title="3D Interactive Plot"):
    data = torch.tensor(data).float()
    latent = model.encoder(data).detach().numpy()
    if isinstance(cmap, str):
        cmap_func = plt.cm.get_cmap(cmap)
        unique_labels = np.unique(labels)
        norm = plt.Normalize(min(unique_labels), max(unique_labels))
        colors = [f'rgb({int(255*r)},{int(255*g)},{int(255*b)})' 
                  for r, g, b, _ in cmap_func(norm(labels))]
    else:
        colors = labels

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=latent[:, 0],
        y=latent[:, 1],
        z=latent[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
            opacity=0.8
        ),
        text=[f"Point {i}" for i in range(len(latent))],
        hoverinfo='text'
    ))

    fig.update_layout(
        title=title,
        width=figsize[0] * 100,
        height=figsize[1] * 100,
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=''),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=''),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=''),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


def get_plot_range_and_center_2D(X):
    max_range = np.array([X[:, 0].ptp(), X[:, 1].ptp()]).max() / 2.0
    mid_x = (X[:, 0].max() + X[:, 0].min()) * 0.5
    mid_y = (X[:, 1].max() + X[:, 1].min()) * 0.5
    return max_range, mid_x, mid_y


def get_plot_range_and_center_3D(X):
    max_range = np.array([X[:, 0].ptp(), X[:, 1].ptp(), X[:, 2].ptp()]).max() / 2.0
    mid_x = (X[:, 0].max() + X[:, 0].min()) * 0.5
    mid_y = (X[:, 1].max() + X[:, 1].min()) * 0.5
    mid_z = (X[:, 2].max() + X[:, 2].min()) * 0.5
    return max_range, mid_x, mid_y, mid_z


def plot2D(fig, X, color, position, title, s=1):
    max_range, mid_x, mid_y = get_plot_range_and_center_2D(X)

    # equalize the aspect ratio of ax1
    ax = fig.add_subplot(position)
    ax.set_aspect('equal', adjustable='box')
    ax.scatter(X[:, 0], X[:, 1], c=color, s=s)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(mid_x - 1.1 * max_range, mid_x + 1.1 * max_range)
    ax.set_ylim(mid_y - 1.1 * max_range, mid_y + 1.1 * max_range)
    ax.set_title(title)


def plot3D(fig, X, color, position, title, s=1):
    max_range, mid_x, mid_y, mid_z = get_plot_range_and_center_3D(X)

    ax = fig.add_subplot(position, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, s=s)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_xlim(mid_x - 1.1 * max_range, mid_x + 1.1 * max_range)
    ax.set_ylim(mid_y - 1.1 * max_range, mid_y + 1.1 * max_range)
    ax.set_zlim(mid_z - 1.1 * max_range, mid_z + 1.1 * max_range)


def visualization(X, autoencoder, color, s=1):
    fig = plt.figure(figsize=(8, 8))
    lat = autoencoder.encoder(torch.from_numpy(X).float()).detach().numpy()
    recon = autoencoder.decoder(torch.from_numpy(lat).float()).detach().numpy()

    plot2D(fig, lat, color, 221, 'AE 2D Latent Space', s)

    if lat.shape[1] == 3:
        plot3D(fig, lat, color, 222, 'AE 3D Latent Data', s)

    if recon.shape[1] == 3:
        plot3D(fig, recon, color, 223, 'AE Reconstructed Data', s)

    if X.shape[1] == 3:
        plot3D(fig, X, color, 224, 'Original Data', s)

    plt.show()


def visualizations(model, data, labels, indices=None, method_name="", recon=False, connect=False, figsize=(5, 5), cmap='viridis'):
    """
    Visualize the latent space and optionally the reconstructed data from an autoencoder.
    
    Args:
        model: The autoencoder model to visualize
        data: Input data to encode and optionally reconstruct
        labels: Color labels for the data points
        indices: Indices for connecting points in the visualization (used if connect=True)
        method_name: Name of the method for saving the figure
        recon: Whether to visualize reconstructed data
        connect: Whether to connect points according to indices
        figsize: Figure size as a tuple (width, height)
        cmap: Colormap for visualization
        
    Returns:
        None
    """
    # Create a figure for the latent space visualization
    fig = plt.figure(figsize=figsize)

    # Convert data to tensor and get latent and reconstructed representations
    data = torch.tensor(data).float()
    latent = model.encoder(data).detach().numpy()
    reconstructed = model(data).detach().numpy()

    # 2D latent space visualization
    if latent.shape[1] == 2:
        plt.scatter(latent[:, 0], latent[:, 1], c=labels, cmap=cmap, s=1)
        plt.grid(False)
        plt.axis('off')

        # Optionally connect points according to indices
        if connect and indices is not None:
            for i in range(len(data)):
                for j in indices[i]:
                    plt.plot([latent[i, 0], latent[j, 0]], 
                             [latent[i, 1], latent[j, 1]], 
                             c='gray', alpha=0.1)

        plt.axis('equal')
        plt.tight_layout()

        # Save figure if method_name is provided
        if method_name:
            plt.savefig(f'{method_name}_latent_space.pdf', format='pdf', bbox_inches='tight')

        plt.show()

    # 3D latent space visualization
    elif latent.shape[1] == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2], 
                   c=labels, cmap=cmap, s=10)
        plt.grid(False)

        # Optionally connect points in 3D
        if connect and indices is not None:
            for i in range(len(data)):
                for j in indices[i]:
                    ax.plot([latent[i, 0], latent[j, 0]], 
                            [latent[i, 1], latent[j, 1]],
                            [latent[i, 2], latent[j, 2]],
                            c='gray', alpha=0.1)

        # ax.set_axis_off()
        ax.grid(False)
        ax.axis('equal')
        plt.tight_layout()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Save figure if method_name is provided
        if method_name:
            plt.savefig(f'swissroll_{method_name}_latent_space_3d.pdf', 
                        format='pdf', bbox_inches='tight')

        plt.show()

    # Reconstructed data visualization (if requested)
    if recon and reconstructed.shape[1] >= 2:
        fig = plt.figure(figsize=figsize)

        # 3D reconstruction visualization
        if reconstructed.shape[1] >= 3:
            ax = fig.add_subplot(projection='3d')
            ax.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], 
                       c=labels, cmap=cmap, s=10)
            ax.grid(False)
            ax.set_axis_off()

            # Save figure if method_name is provided
            if method_name:
                plt.savefig(f'swissroll_{method_name}_reconstruction.pdf', 
                            format='pdf', bbox_inches='tight')

            plt.show()
        # 2D reconstruction visualization
        else:
            plt.scatter(reconstructed[:, 0], reconstructed[:, 1], 
                        c=labels, cmap=cmap, s=10)
            plt.grid(False)
            plt.axis('off')
            plt.axis('equal')

            # Save figure if method_name is provided
            if method_name:
                plt.savefig(f'swissroll_{method_name}_reconstruction.pdf', 
                            format='pdf', bbox_inches='tight')

            plt.show()

    return None


def visualize_test_reconstruction_and_latent(encoder, decoder, test_loader, device='cpu'):
    """
    1) Take the first test batch -> reconstruct -> show side-by-side original vs reconstructed
    2) Gather entire test set -> plot 2D latent scatter (if latent_dim=2).
    """

    encoder.eval()
    decoder.eval()

    # -------------------------------------------------------
    # Part A: Show reconstruction from the first test batch
    # -------------------------------------------------------
    with torch.no_grad():
        batch = next(iter(test_loader))
        test_imgs, color, idxs = batch  # e.g. shape (B, 3, H, W)
        test_imgs = test_imgs.to(device)

        z_test = encoder(test_imgs)
        x_recon = decoder(z_test)

        # Show side-by-side using make_grid
        orig_grid = make_grid(test_imgs.cpu(), nrow=4, pad_value=1)
        recon_grid = make_grid(x_recon.cpu(), nrow=4, pad_value=1)

        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        axs[0].imshow(np.transpose(orig_grid.numpy(), (1, 2, 0)))
        axs[0].set_title("Original")
        axs[0].axis('off')

        axs[1].imshow(np.transpose(recon_grid.numpy(), (1, 2, 0)))
        axs[1].set_title("Reconstructed")
        axs[1].axis('off')
        plt.show()

    # -------------------------------------------------------
    # Part B: Plot the entire test set in latent space (2D)
    # -------------------------------------------------------
    test_data_list, test_color_list = [], []
    for images, color_batch, _ in test_loader:
        test_data_list.append(images)
        test_color_list.append(color_batch)

    # shape: (N, C, H, W)
    X_test = torch.cat(test_data_list, dim=0)
    color_test = torch.cat(test_color_list, dim=0)

    with torch.no_grad():
        # run entire test set through encoder -> decoder
        X_test_gpu = X_test.to(device)
        z_all = encoder(X_test_gpu.float())
        x_all = decoder(z_all)

    # Convert to numpy for plotting
    lat_all = z_all.cpu().numpy()                # shape (N, latent_dim)
    recon_all = x_all.cpu().numpy()              # shape (N, C, H, W) or (N, d)
    color_all = color_test.cpu().numpy()        # shape (N,)

    # If your latent dimension is 2, do a scatter plot
    if lat_all.shape[1] == 2:
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(121)
        sc = ax1.scatter(lat_all[:, 0], lat_all[:, 1], c=color_all, cmap='jet')
        ax1.set_xlabel('Latent Dim 1')
        ax1.set_ylabel('Latent Dim 2')
        ax1.set_title('AE 2D Latent Space')
        # plt.colorbar(sc, ax=ax1)
        plt.show()
    else:
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(121)
        sc = ax1.scatter(lat_all[:, 0], lat_all[:, 1], lat_all[:, 2], c=color_all, cmap='jet')
        # print(f"Latent dimension = {lat_all.shape[1]}, not 2D. Skipping scatter plot.")


def plot_distance_histogram(GeoD):
    distances = GeoD.flatten()

    # remove diagonal elements
    n = GeoD.shape[0]
    mask = ~np.eye(n, dtype=bool).flatten()
    distances = distances[mask]

    # plot histogram
    plt.figure(figsize=(5, 3))
    plt.hist(distances, bins=50, edgecolor='black')
    plt.title('distribution of pairwise distances')
    plt.xlabel('distance')
    plt.ylabel('number of pairs')
    plt.grid(True)
    plt.show()