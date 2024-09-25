import os
import threading
import time
from typing import Dict, Optional

import torch
from torch.nn import Module

import plotly.express as px
from plotly.graph_objs import Figure

import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox


def create_visualization_directory(directory: str) -> None:
    """
    Creates the visualization directory if it does not exist.

    Args:
        directory (str): Path to the directory.
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create directory '{directory}': {e}") from e


def save_plot(fig: Figure, filepath: str) -> None:
    """
    Saves the Plotly figure to an HTML file.

    Args:
        fig (Figure): Plotly figure object.
        filepath (str): Path to save the HTML file.
    """
    try:
        fig.write_html(filepath, include_plotlyjs='cdn', full_html=True)
    except Exception as e:
        raise RuntimeError(f"Failed to save plot to '{filepath}': {e}") from e


def _prepare_data(param_tensor: torch.Tensor) -> torch.Tensor:
    """
    Prepares the tensor data by detaching and moving it to CPU.

    Args:
        param_tensor (torch.Tensor): The parameter tensor.

    Returns:
        torch.Tensor: Prepared tensor.
    """
    return param_tensor.detach().cpu()


def _get_plot_title(param_name: str, tensor_shape: torch.Size, dimensions: int) -> str:
    """
    Generates a plot title based on parameter name, shape, and dimensions.

    Args:
        param_name (str): Name of the parameter.
        tensor_shape (torch.Size): Shape of the tensor.
        dimensions (int): Number of dimensions.

    Returns:
        str: Generated title.
    """
    return f"{dimensions}D Plot for '{param_name}' | Shape: {tensor_shape}"


def _get_labels(dimensions: int) -> Dict[str, str]:
    """
    Generates labels for the plot axes based on the number of dimensions.

    Args:
        dimensions (int): Number of dimensions.

    Returns:
        Dict[str, str]: Axis labels.
    """
    labels = {}
    for i in range(1, dimensions + 1):
        labels_key = f"Dimension {i}"
        labels[f"Dim{i}"] = labels_key
    return labels


def _apply_layout(
    fig: Figure,
    title: str,
    labels: Dict[str, str],
    template: str = "plotly_dark"
) -> None:
    """
    Applies layout settings to the Plotly figure for a professional look.

    Args:
        fig (Figure): The Plotly figure to modify.
        title (str): Title of the plot.
        labels (Dict[str, str]): Axis labels.
        template (str, optional): Plotly template. Defaults to "plotly_dark".
    """
    fig.update_layout(
        title=title,
        title_x=0.5,
        template=template,
        legend_title_text="Legend",
        title_font=dict(size=20, family="Arial"),
        font=dict(size=14, family="Arial"),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))


def plot_1d(param_name: str, param_tensor: torch.Tensor, output_dir: str) -> str:
    """
    Creates and saves a box plot for a 1D tensor.

    Args:
        param_name (str): Name of the parameter.
        param_tensor (torch.Tensor): 1D tensor.
        output_dir (str): Directory to save the plot.

    Returns:
        str: Filepath of the saved plot.
    """
    data = {
        'Values': _prepare_data(param_tensor).numpy()
    }
    title = _get_plot_title(param_name, param_tensor.shape, dimensions=1)
    fig = px.box(
        data,
        y='Values',
        title=title,
        labels={'Values': 'Parameter Values'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    _apply_layout(fig, title, labels={'Values': 'Parameter Values'})
    filepath = os.path.join(output_dir, f"{param_name}_box_plot.html")
    save_plot(fig, filepath)
    return filepath


def plot_2d(param_name: str, param_tensor: torch.Tensor, output_dir: str) -> str:
    """
    Creates and saves a scatter plot for a 2D tensor.

    Args:
        param_name (str): Name of the parameter.
        param_tensor (torch.Tensor): 2D tensor.
        output_dir (str): Directory to save the plot.

    Returns:
        str: Filepath of the saved plot.
    """
    if param_tensor.shape[0] < 2:
        raise ValueError(
            f"Cannot plot 2D parameter '{param_name}': requires at least two elements in the first dimension."
        )

    reshaped = _prepare_data(param_tensor).numpy()
    data = {
        'Dim1': reshaped[:, 0],
        'Dim2': reshaped[:, 1]
    }
    title = _get_plot_title(param_name, param_tensor.shape, dimensions=2)
    fig = px.scatter(
        data,
        x='Dim1',
        y='Dim2',
        title=title,
        labels={'Dim1': 'Dimension 1', 'Dim2': 'Dimension 2'},
        color_discrete_sequence=px.colors.qualitative.T10
    )
    _apply_layout(fig, title, labels={'Dim1': 'Dimension 1', 'Dim2': 'Dimension 2'})
    filepath = os.path.join(output_dir, f"{param_name}_scatter_plot.html")
    save_plot(fig, filepath)
    return filepath


def plot_3d(param_name: str, param_tensor: torch.Tensor, output_dir: str) -> str:
    """
    Creates and saves a 3D scatter plot for a 3D tensor.

    Args:
        param_name (str): Name of the parameter.
        param_tensor (torch.Tensor): 3D tensor.
        output_dir (str): Directory to save the plot.

    Returns:
        str: Filepath of the saved plot.
    """
    reshaped = _prepare_data(param_tensor).reshape(-1, 3).numpy()
    data = {
        'Dim1': reshaped[:, 0],
        'Dim2': reshaped[:, 1],
        'Dim3': reshaped[:, 2]
    }
    title = _get_plot_title(param_name, param_tensor.shape, dimensions=3)
    fig = px.scatter_3d(
        data,
        x='Dim1',
        y='Dim2',
        z='Dim3',
        title=title,
        labels={
            'Dim1': 'Dimension 1',
            'Dim2': 'Dimension 2',
            'Dim3': 'Dimension 3'
        },
        color='Dim3',
        color_continuous_scale='Viridis'
    )
    _apply_layout(fig, title, labels={
        'Dim1': 'Dimension 1',
        'Dim2': 'Dimension 2',
        'Dim3': 'Dimension 3'
    })
    filepath = os.path.join(output_dir, f"{param_name}_3d_scatter_plot.html")
    save_plot(fig, filepath)
    return filepath


def plot_4d(param_name: str, param_tensor: torch.Tensor, output_dir: str) -> str:
    """
    Creates and saves a 3D scatter plot with color representing the 4th dimension for a 4D tensor.

    Args:
        param_name (str): Name of the parameter.
        param_tensor (torch.Tensor): 4D tensor.
        output_dir (str): Directory to save the plot.

    Returns:
        str: Filepath of the saved plot.
    """
    reshaped = _prepare_data(param_tensor).reshape(-1, 4).numpy()
    data = {
        'Dim1': reshaped[:, 0],
        'Dim2': reshaped[:, 1],
        'Dim3': reshaped[:, 2],
        'Dim4': reshaped[:, 3]
    }
    title = _get_plot_title(param_name, param_tensor.shape, dimensions=4)
    fig = px.scatter_3d(
        data,
        x='Dim1',
        y='Dim2',
        z='Dim3',
        color='Dim4',
        title=title,
        labels={
            'Dim1': 'Dimension 1',
            'Dim2': 'Dimension 2',
            'Dim3': 'Dimension 3',
            'Dim4': 'Dimension 4'
        },
        color_continuous_scale='Plasma'
    )
    _apply_layout(fig, title, labels={
        'Dim1': 'Dimension 1',
        'Dim2': 'Dimension 2',
        'Dim3': 'Dimension 3',
        'Dim4': 'Dimension 4'
    })
    filepath = os.path.join(output_dir, f"{param_name}_4d_scatter_plot.html")
    save_plot(fig, filepath)
    return filepath


def plot_5d(param_name: str, param_tensor: torch.Tensor, output_dir: str) -> str:
    """
    Creates and saves a 3D scatter plot with color and size representing the 4th and 5th dimensions for a 5D tensor.

    Args:
        param_name (str): Name of the parameter.
        param_tensor (torch.Tensor): 5D tensor.
        output_dir (str): Directory to save the plot.

    Returns:
        str: Filepath of the saved plot.
    """
    reshaped = _prepare_data(param_tensor).reshape(-1, 5).numpy()
    data = {
        'Dim1': reshaped[:, 0],
        'Dim2': reshaped[:, 1],
        'Dim3': reshaped[:, 2],
        'Dim4': reshaped[:, 3],
        'Dim5': reshaped[:, 4]
    }
    title = _get_plot_title(param_name, param_tensor.shape, dimensions=5)
    fig = px.scatter_3d(
        data,
        x='Dim1',
        y='Dim2',
        z='Dim3',
        color='Dim4',
        size='Dim5',
        title=title,
        labels={
            'Dim1': 'Dimension 1',
            'Dim2': 'Dimension 2',
            'Dim3': 'Dimension 3',
            'Dim4': 'Dimension 4',
            'Dim5': 'Dimension 5'
        },
        color_continuous_scale='Cividis',
        size_max=18
    )
    _apply_layout(fig, title, labels={
        'Dim1': 'Dimension 1',
        'Dim2': 'Dimension 2',
        'Dim3': 'Dimension 3',
        'Dim4': 'Dimension 4',
        'Dim5': 'Dimension 5'
    })
    filepath = os.path.join(output_dir, f"{param_name}_5d_scatter_plot.html")
    save_plot(fig, filepath)
    return filepath


def visualize_model_parameters(model: Module, output_dir: str = "model_parameter_plots") -> None:
    """
    Visualizes parameters of a PyTorch model based on their dimensionality.

    Args:
        model (Module): The PyTorch model.
        output_dir (str, optional): Directory to save the plots. Defaults to "model_parameter_plots".
    """
    create_visualization_directory(output_dir)

    plot_dispatcher = {
        1: plot_1d,
        2: plot_2d,
        3: plot_3d,
        4: plot_4d,
        5: plot_5d
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip non-trainable parameters

        ndim = param.ndimension()

        plot_func = plot_dispatcher.get(ndim)
        if plot_func:
            try:
                filepath = plot_func(name, param, output_dir)
                yield filepath, name
            except Exception as e:
                print(f"Error plotting {ndim}D parameter '{name}': {e}")
        else:
            print(f"Skipping parameter '{name}' with unsupported dimensions: {ndim}D")


class VisualizationController:
    """
    Controls the visualization process, managing the GUI and sequential plotting.
    """

    def __init__(self, model: Module, output_dir: str = "model_parameter_plots", delay: float = 2.0):
        """
        Initializes the VisualizationController.

        Args:
            model (Module): The PyTorch model to visualize.
            output_dir (str, optional): Directory to save the plots. Defaults to "model_parameter_plots".
            delay (float, optional): Delay between visualizations in seconds. Defaults to 2.0.
        """
        self.model = model
        self.output_dir = output_dir
        self.delay = delay
        self.is_playing = False
        self.thread: Optional[threading.Thread] = None
        self.plot_generator = visualize_model_parameters(self.model, self.output_dir)

        self._setup_gui()

    def _setup_gui(self) -> None:
        """
        Sets up the Tkinter GUI.
        """
        self.root = tk.Tk()
        self.root.title("Model Parameter Visualization")
        self.root.geometry("300x150")
        self.root.resizable(False, False)

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        self.play_button = ttk.Button(main_frame, text="Play", command=self.start_visualization)
        self.play_button.pack(pady=10)

        self.stop_button = ttk.Button(main_frame, text="Stop", command=self.stop_visualization, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.status_label = ttk.Label(main_frame, text="Status: Ready")
        self.status_label.pack(pady=10)

    def start_visualization(self) -> None:
        """
        Starts the visualization process in a separate thread.
        """
        if not self.is_playing:
            self.is_playing = True
            self.play_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Playing")
            self.thread = threading.Thread(target=self._visualization_loop, daemon=True)
            self.thread.start()

    def stop_visualization(self) -> None:
        """
        Stops the visualization process.
        """
        if self.is_playing:
            self.is_playing = False
            self.play_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Stopped")

    def _visualization_loop(self) -> None:
        """
        The main loop for visualization, running in a separate thread.
        """
        try:
            for filepath, name in self.plot_generator:
                if not self.is_playing:
                    break
                self.status_label.config(text=f"Visualizing: {name}")
                webbrowser.open(f'file://{os.path.abspath(filepath)}')
                time.sleep(self.delay)
            self.stop_visualization()
            self.status_label.config(text="Status: Completed")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during visualization: {e}")
            self.stop_visualization()

    def run(self) -> None:
        """
        Runs the Tkinter main loop.
        """
        self.root.mainloop()


def main() -> None:
    """
    The main function to instantiate the model and start the visualization controller.
    """
    try:
        from transformers import AutoModelForCausalLM

        # Instantiate the model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as e:
        print(f"Failed to load the model: {e}")
        return

    # Initialize and run the visualization controller
    controller = VisualizationController(model=model, output_dir="model_parameter_plots", delay=2.0)
    controller.run()


if __name__ == "__main__":
    main()