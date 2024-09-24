import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import plotly.graph_objs as go
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
from transformers import AutoModelForCausalLM
import logging
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np
import plotly.graph_objs as go
import torch
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelVisualizer:
    def __init__(self, output_dir: Union[str, Path] = "visualizations"):
        self.output_dir = Path(output_dir)
        self.html_dir = self.output_dir / "html"
        self.png_dir = self.output_dir / "png"
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.png_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizations will be saved in: {self.output_dir}")

    def visualize_tensor(self, tensor: torch.Tensor, metadata: Dict[str, Any]) -> None:
        dim = tensor.dim()
        shape = tensor.shape
        logger.info(f"Visualizing a tensor with shape: {shape}")

        try:
            if dim == 1:
                self._plot_1d(tensor, metadata)
            elif dim == 2:
                self._plot_2d(tensor, metadata)
            elif dim == 3:
                self._plot_3d(tensor, metadata)
            elif dim == 4:
                self._plot_4d(tensor, metadata)
            else:
                self._plot_heatmap(tensor, metadata)
        except Exception as e:
            logger.error(f"Error in visualizing tensor: {e}", exc_info=True)

    def _plot_1d(self, tensor: torch.Tensor, metadata: Dict[str, Any]) -> None:
        tensor = tensor.cpu().squeeze()
        x = np.arange(tensor.size(0))
        y = tensor.numpy()

        fig = go.Figure(go.Scatter(x=x, y=y, mode='lines+markers'))
        title = f"1D Plot: {metadata.get('layer_name', 'Unknown Layer')}"
        fig.update_layout(title=title, xaxis_title='Index', yaxis_title='Value')
        self._save_and_export(fig, f"{metadata.get('layer_name', 'layer')}_1d_plot")

    def _plot_2d(self, tensor: torch.Tensor, metadata: Dict[str, Any]) -> None:
        tensor = tensor.cpu().squeeze()
        fig = go.Figure(data=[go.Heatmap(z=tensor.numpy())])
        title = f"2D Heatmap: {metadata.get('layer_name', 'Unknown Layer')}"
        fig.update_layout(title=title)
        self._save_and_export(fig, f"{metadata.get('layer_name', 'layer')}_2d_heatmap")

    def _plot_3d(self, tensor: torch.Tensor, metadata: Dict[str, Any]) -> None:
        tensor = tensor.cpu().squeeze()
        if tensor.dim() == 3:
            fig = go.Figure(data=[go.Surface(z=tensor.numpy())])
            title = f"3D Surface: {metadata.get('layer_name', 'Unknown Layer')}"
            fig.update_layout(title=title, scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        else:
            fig = self._plot_heatmap(tensor, metadata)
        self._save_and_export(fig, f"{metadata.get('layer_name', 'layer')}_3d_plot")

    def _plot_4d(self, tensor: torch.Tensor, metadata: Dict[str, Any]) -> None:
        tensor = tensor.cpu().squeeze()
        if tensor.dim() == 4:
            fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'surface'}]*2]*2,
                                subplot_titles=[f"Channel {i}" for i in range(4)])
            for i in range(4):
                row = i // 2 + 1
                col = i % 2 + 1
                fig.add_trace(go.Surface(z=tensor[i].numpy()), row=row, col=col)
            title = f"4D Plot: {metadata.get('layer_name', 'Unknown Layer')}"
            fig.update_layout(title=title, height=800)
        else:
            fig = self._plot_heatmap(tensor, metadata)
        self._save_and_export(fig, f"{metadata.get('layer_name', 'layer')}_4d_plot")

    def _plot_heatmap(self, tensor: torch.Tensor, metadata: Dict[str, Any]) -> go.Figure:
        tensor = tensor.cpu().squeeze()
        if tensor.dim() > 2:
            tensor = tensor.mean(dim=tuple(range(tensor.dim() - 2)))
        fig = go.Figure(data=[go.Heatmap(z=tensor.numpy())])
        title = f"Heatmap: {metadata.get('layer_name', 'Unknown Layer')} (Shape: {tuple(tensor.shape)})"
        fig.update_layout(title=title)
        return fig

    def _save_and_export(self, fig: go.Figure, name: str) -> None:
        fig.write_html(str(self.html_dir / f"{name}.html"))
        fig.write_image(str(self.png_dir / f"{name}.png"))

    def _generate_title(self, metadata: Dict[str, Any], plot_type: str) -> str:
        return f"{plot_type}: {metadata.get('layer_name', 'Unknown Layer')}"

    def _sanitize_filename(self, filename: str) -> str:
        return "".join(c for c in filename if c.isalnum() or c in (' ', '_', '-')).rstrip()

class AdvancedModelWrapper(nn.Module):
    """
    A wrapper class for advanced model visualization.
    """

    def __init__(self, model: nn.Module, visualizer: ModelVisualizer) -> None:
        """
        Initialize the AdvancedModelWrapper.

        Args:
            model (nn.Module): The PyTorch model to wrap.
            visualizer (ModelVisualizer): The visualizer instance.
        """
        super().__init__()
        self.model = model
        self.visualizer = visualizer
        self.hooks: List[Callable] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Register forward hooks for all suitable modules in the model.
        """
        for name, module in self.model.named_modules():
            if self._is_visualizable_module(module):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
                logger.debug(f"Hook registered for layer: {name}")

    @staticmethod
    def _is_visualizable_module(module: nn.Module) -> bool:
        """
        Determine if the module is suitable for visualization.

        Args:
            module (nn.Module): The module to check.

        Returns:
            bool: True if visualizable, False otherwise.
        """
        # You can refine this method to include only specific layer types
        # For now, we'll visualize all layers
        return True

    def _make_hook(self, name: str) -> Callable:
        """
        Create a hook function for the given module name.

        Args:
            name (str): Name of the module.

        Returns:
            Callable: The hook function.
        """

        def hook(module: nn.Module, input: Any, output: Any) -> None:
            metadata: Dict[str, Any] = {
                'model_type': type(self.model).__name__,
                'layer_name': name,
                'layer_type': type(module).__name__
            }
            tensors = self._extract_tensors(output)
            if not tensors:
                logger.warning(f"No tensors found in output of layer {name}.")
                return
            for idx, tensor in enumerate(tensors):
                meta = metadata.copy()
                if len(tensors) > 1:
                    meta['output_index'] = idx
                self.visualizer.visualize_tensor(tensor, meta)

        return hook

    @staticmethod
    def _extract_tensors(output: Any) -> List[torch.Tensor]:
        """
        Recursively extract tensors from complex output structures.

        Args:
            output (Any): The output from a module's forward pass.

        Returns:
            List[torch.Tensor]: A list of tensors extracted from the output.
        """
        tensors: List[torch.Tensor] = []

        if isinstance(output, torch.Tensor):
            tensors.append(output)
        elif isinstance(output, (list, tuple)):
            for item in output:
                tensors.extend(AdvancedModelWrapper._extract_tensors(item))
        elif isinstance(output, dict):
            for item in output.values():
                tensors.extend(AdvancedModelWrapper._extract_tensors(item))
        elif hasattr(output, '__dict__'):
            for item in output.__dict__.values():
                tensors.extend(AdvancedModelWrapper._extract_tensors(item))
        return tensors

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass through the wrapped model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The output of the wrapped model.
        """
        try:
            return self.model(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error during forward pass: {e}", exc_info=True)
            raise

    def remove_hooks(self) -> None:
        """
        Remove all registered hooks.
        """
        for hook in self.hooks:
            hook.remove()
        logger.info("All hooks have been removed.")
        self.hooks.clear()


def visualize_model(
    model: nn.Module,
    input_data: torch.Tensor,
    output_dir: str = "visualizations"
) -> None:
    """
    Visualize the internals of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to visualize.
        input_data (torch.Tensor): Sample input data for the model.
        output_dir (str): Directory to save visualization outputs.
    """
    visualizer = ModelVisualizer(output_dir)
    wrapped_model = AdvancedModelWrapper(model, visualizer)

    try:
        logger.info("Starting model visualization...")
        wrapped_model.eval()
        with torch.no_grad():
            wrapped_model(input_data)
        logger.info("Model visualization completed successfully.")
    except Exception as e:
        logger.error(f"Error during model visualization: {e}", exc_info=True)
    finally:
        wrapped_model.remove_hooks()


def main() -> None:
    """
    Main function to demonstrate model visualization.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return

    # Create sample input data
    # Ensure that the input data matches the model's expected input dimensions and type
    try:
        input_tensor = torch.randint(
            low=0,
            high=50257,
            size=(10, 10),  # Changed to batch_size=1, sequence_length=10 for efficiency
            dtype=torch.long
        )
        logger.info("Input tensor created successfully.")
    except Exception as e:
        logger.error(f"Failed to create input tensor: {e}", exc_info=True)
        return

    # Visualize the model
    visualize_model(model, input_tensor)


if __name__ == "__main__":
    main()