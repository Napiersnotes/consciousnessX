"""
Neural network visualization utilities for ConsciousnessX.

Visualize neural activity, connectivity, and dynamics.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap


class NeuralVisualizer:
    """
    Visualize neural network activity and dynamics.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize neural visualizer.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        self.figures = {}
        self.axes = {}
    
    def plot_activity_heatmap(
        self,
        activity: np.ndarray,
        time_points: Optional[np.ndarray] = None,
        title: str = "Neural Activity",
        colormap: str = 'viridis',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot neural activity as heatmap over time.
        
        Args:
            activity: Activity matrix (neurons x time)
            time_points: Optional time points
            title: Plot title
            colormap: Colormap name
            save_path: Optional path to save figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        extent = [0, activity.shape[1], 0, activity.shape[0]]
        if time_points is not None:
            extent[0] = time_points[0]
            extent[1] = time_points[-1]
        
        im = ax.imshow(activity, aspect='auto', cmap=colormap, 
                      origin='lower', extent=extent, interpolation='nearest')
        
        plt.colorbar(im, ax=ax, label='Activity')
        ax.set_xlabel('Time')
        ax.set_ylabel('Neuron')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures['activity_heatmap'] = fig
        return fig
    
    def plot_raster_plot(
        self,
        spike_times: List[np.ndarray],
        neuron_ids: Optional[np.ndarray] = None,
        title: str = "Spike Raster",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot spike raster plot.
        
        Args:
            spike_times: List of spike time arrays for each neuron
            neuron_ids: Optional neuron IDs
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        num_neurons = len(spike_times)
        if neuron_ids is None:
            neuron_ids = np.arange(num_neurons)
        
        for i, spikes in enumerate(spike_times):
            ax.scatter(spikes, [i] * len(spikes), s=1, color='black', alpha=0.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Neuron')
        ax.set_title(title)
        ax.set_yticks(range(num_neurons))
        ax.set_yticklabels(neuron_ids)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures['raster_plot'] = fig
        return fig
    
    def plot_connectivity_matrix(
        self,
        weights: np.ndarray,
        title: str = "Neural Connectivity",
        colormap: str = 'coolwarm',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot neural connectivity weight matrix.
        
        Args:
            weights: Weight matrix (neurons x neurons)
            title: Plot title
            colormap: Colormap name
            save_path: Optional path to save figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(weights, cmap=colormap, aspect='auto')
        plt.colorbar(im, ax=ax, label='Weight')
        ax.set_xlabel('Source Neuron')
        ax.set_ylabel('Target Neuron')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures['connectivity'] = fig
        return fig
    
    def plot_firing_rates(
        self,
        firing_rates: np.ndarray,
        time_points: Optional[np.ndarray] = None,
        title: str = "Firing Rates",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot firing rates over time.
        
        Args:
            firing_rates: Firing rate array (time,)
            time_points: Optional time points
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if time_points is None:
            time_points = np.arange(len(firing_rates))
        
        ax.plot(time_points, firing_rates, linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures['firing_rates'] = fig
        return fig
    
    def plot_neural_trajectory(
        self,
        activity: np.ndarray,
        n_components: int = 2,
        title: str = "Neural Trajectory",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot neural state trajectory using PCA.
        
        Args:
            activity: Activity matrix (neurons x time)
            n_components: Number of PCA components (2 or 3)
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Figure object
        """
        from sklearn.decomposition import PCA
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        activity_pca = pca.fit_transform(activity.T)
        
        if n_components == 2:
            fig, ax = plt.subplots(figsize=self.figsize)
            scatter = ax.scatter(activity_pca[:, 0], activity_pca[:, 1], 
                               c=np.arange(activity_pca.shape[0]), 
                               cmap='viridis', alpha=0.6)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.colorbar(scatter, ax=ax, label='Time')
        elif n_components == 3:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(activity_pca[:, 0], activity_pca[:, 1], 
                               activity_pca[:, 2],
                               c=np.arange(activity_pca.shape[0]), 
                               cmap='viridis', alpha=0.6)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
            plt.colorbar(scatter, ax=ax, label='Time')
        else:
            raise ValueError("n_components must be 2 or 3")
        
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures['trajectory'] = fig
        return fig
    
    def animate_activity(
        self,
        activity: np.ndarray,
        interval: int = 50,
        save_path: Optional[str] = None
    ) -> animation.FuncAnimation:
        """
        Create animation of neural activity.
        
        Args:
            activity: Activity matrix (neurons x time)
            interval: Frame interval in ms
            save_path: Optional path to save animation
            
        Returns:
            Animation object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create grid for visualization
        grid_size = int(np.ceil(np.sqrt(activity.shape[0])))
        reshaped_activity = np.zeros((grid_size, grid_size, activity.shape[1]))
        reshaped_activity[:activity.shape[0], :activity.shape[0], :] = activity.reshape(
            grid_size, grid_size, -1
        )[:activity.shape[0], :activity.shape[0], :]
        
        # Initial frame
        im = ax.imshow(reshaped_activity[:, :, 0], cmap='viridis', 
                      vmin=activity.min(), vmax=activity.max())
        plt.colorbar(im, ax=ax, label='Activity')
        ax.set_title('Neural Activity')
        
        def update(frame):
            im.set_array(reshaped_activity[:, :, frame])
            ax.set_title(f'Neural Activity - Frame {frame}')
            return [im]
        
        anim = animation.FuncAnimation(
            fig, update, frames=activity.shape[1], 
            interval=interval, blit=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1000//interval)
        
        return anim
    
    def plot_phi_dynamics(
        self,
        phi_values: np.ndarray,
        collapse_events: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        title: str = "Phi Dynamics",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot integrated information (phi) dynamics.
        
        Args:
            phi_values: Phi values over time
            collapse_events: Optional collapse event times
            threshold: Optional phi threshold for consciousness
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        time_points = np.arange(len(phi_values))
        ax.plot(time_points, phi_values, linewidth=2, label='Phi')
        
        if threshold is not None:
            ax.axhline(threshold, color='red', linestyle='--', 
                      label=f'Threshold ({threshold})')
        
        if collapse_events is not None:
            ax.scatter(collapse_events, phi_values[collapse_events.astype(int)], 
                      color='red', s=50, marker='x', label='Collapse Events', zorder=5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Phi')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures['phi_dynamics'] = fig
        return fig
    
    def close_all(self) -> None:
        """Close all figures."""
        plt.close('all')
        self.figures.clear()
        self.axes.clear()