import os
from typing import Dict

import torch
from torch import nn
from torch.nn import Module

import plotly.express as px
from plotly.graph_objs import Figure


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
        fig.write_html(filepath, include_plotlyjs='cdn', full_html=False)
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
    dimension_str = "D" if dimensions > 1 else "D"
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
    fig: Figure, title: str, labels: Dict[str, str], template: str = "plotly_dark"
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


def plot_1d(param_name: str, param_tensor: torch.Tensor, output_dir: str) -> None:
    """
    Creates and saves a box plot for a 1D tensor.

    Args:
        param_name (str): Name of the parameter.
        param_tensor (torch.Tensor): 1D tensor.
        output_dir (str): Directory to save the plot.
    """
    try:
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
    except Exception as e:
        print(f"Error plotting 1D parameter '{param_name}': {e}")


def plot_2d(param_name: str, param_tensor: torch.Tensor, output_dir: str) -> None:
    """
    Creates and saves a scatter plot for a 2D tensor.

    Args:
        param_name (str): Name of the parameter.
        param_tensor (torch.Tensor): 2D tensor.
        output_dir (str): Directory to save the plot.
    """
    if param_tensor.shape[0] < 2:
        print(f"Cannot plot 2D parameter '{param_name}': requires at least two elements in the first dimension.")
        return

    try:
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
    except Exception as e:
        print(f"Error plotting 2D parameter '{param_name}': {e}")


def plot_3d(param_name: str, param_tensor: torch.Tensor, output_dir: str) -> None:
    """
    Creates and saves a 3D scatter plot for a 3D tensor.

    Args:
        param_name (str): Name of the parameter.
        param_tensor (torch.Tensor): 3D tensor.
        output_dir (str): Directory to save the plot.
    """
    try:
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
    except Exception as e:
        print(f"Error plotting 3D parameter '{param_name}': {e}")


def plot_4d(param_name: str, param_tensor: torch.Tensor, output_dir: str) -> None:
    """
    Creates and saves a 3D scatter plot with color representing the 4th dimension for a 4D tensor.

    Args:
        param_name (str): Name of the parameter.
        param_tensor (torch.Tensor): 4D tensor.
        output_dir (str): Directory to save the plot.
    """
    try:
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
    except Exception as e:
        print(f"Error plotting 4D parameter '{param_name}': {e}")


def plot_5d(param_name: str, param_tensor: torch.Tensor, output_dir: str) -> None:
    """
    Creates and saves a 3D scatter plot with color and size representing the 4th and 5th dimensions for a 5D tensor.

    Args:
        param_name (str): Name of the parameter.
        param_tensor (torch.Tensor): 5D tensor.
        output_dir (str): Directory to save the plot.
    """
    try:
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
    except Exception as e:
        print(f"Error plotting 5D parameter '{param_name}': {e}")


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
            plot_func(name, param, output_dir)
        else:
            print(f"Skipping parameter '{name}' with unsupported dimensions: {ndim}D")


# # Example Usage
# if __name__ == "__main__":
#     from transformers import AutoModelForCausalLM

#     # Instantiate the model
#     try:
#         model = AutoModelForCausalLM.from_pretrained("gpt2")
#         visualize_model_parameters(model, output_dir="model_parameter_plots")
#     except Exception as e:
#         print(f"Failed to visualize model parameters: {e}")



import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenVisualizer:
    """
    A class to visualize the top k probable token generations from a given token ID
    using a specified language model.

    Attributes:
        model_name (str): The name or path of the pre-trained language model.
        top_k (int): The number of top probable tokens to display.
    """

    def __init__(self, model_name: str = "gpt2", top_k: int = 10) -> None:
        """
        Initializes the TokenVisualizer with the specified model and top_k value.

        Args:
            model_name (str): The name or path of the pre-trained language model.
            top_k (int): The number of top probable tokens to display.
        """
        self.model_name: str = model_name
        self.top_k: int = top_k
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        """
        Loads the pre-trained model and tokenizer.
        """
        try:
            logger.info(f"Loading model '{self.model_name}'...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            logger.info("Model loaded successfully.")

            logger.info(f"Loading tokenizer for model '{self.model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise

    def get_token_distribution(self, token_id: int) -> Tuple[List[str], List[float]]:
        """
        Retrieves the top k token probabilities and their corresponding tokens for a given token ID.

        Args:
            token_id (int): The input token ID.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing a list of token strings and their probabilities.
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer not loaded.")
            raise ValueError("Model or tokenizer not loaded.")

        try:
            logger.info(f"Processing token ID: {token_id}")
            input_ids = torch.tensor([[token_id]]).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]  # Get logits for the last token
                probabilities = torch.softmax(logits, dim=-1).squeeze()

            top_probs, top_ids = torch.topk(probabilities, self.top_k)
            tokens = [self.tokenizer.decode([token_id]) for token_id in top_ids.tolist()]
            probs = top_probs.cpu().tolist()

            logger.info(f"Top {self.top_k} tokens retrieved successfully.")
            return tokens, probs
        except Exception as e:
            logger.error(f"Error retrieving token distribution: {e}")
            raise

    def visualize_distribution(
        self, token_id: int, title: Optional[str] = None
    ) -> None:
        """
        Visualizes the token distribution as a bar chart using Plotly.

        Args:
            token_id (int): The input token ID.
            title (Optional[str]): The title of the plot. If None, a default title is used.
        """
        try:
            tokens, probs = self.get_token_distribution(token_id)
            token_texts = [token if token.strip() else "<UNK>" for token in tokens]

            plot_title = title or f"Top {self.top_k} Token Probabilities for Token ID {token_id}"
            fig = make_subplots(
                rows=1,
                cols=1,
                subplot_titles=(plot_title,),
            )

            bar = go.Bar(
                x=token_texts,
                y=probs,
                marker=dict(color='rgba(55, 128, 191, 0.7)'),
                name='Probability',
            )

            fig.add_trace(bar, row=1, col=1)

            fig.update_layout(
                title=plot_title,
                xaxis_title="Tokens",
                yaxis_title="Probability",
                legend_title="Legend",
                bargap=0.2,
                template="plotly_white",
            )

            fig.show()
            logger.info("Visualization created successfully.")
        except Exception as e:
            logger.error(f"Error during visualization: {e}")
            raise


# def main() -> None:
#     """
#     Main function to execute the token visualization.
#     """
#     try:
#         # Initialize the visualizer with desired model and top_k
#         visualizer = TokenVisualizer(model_name="gpt2", top_k=100)

#         # Example token ID input; replace with desired token ID
#         example_token_id = visualizer.tokenizer.encode("Hello world", add_special_tokens=False)[-1]

#         # Visualize the token distribution
#         visualizer.visualize_distribution(token_id=example_token_id, title="Token Distribution Example")
#     except Exception as e:
#         logger.error(f"An error occurred in the main execution: {e}")


# if __name__ == "__main__":
#     main()


import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenVisualizer:
    """
    A class to visualize the top k probable token generations from a given token ID
    using a specified language model.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        top_k: int = 10,
        num_steps: int = 30,
    ) -> None:
        """
        Initializes the TokenVisualizer with the specified model, top_k value, and number of steps.

        Args:
            model_name (str): The name or path of the pre-trained language model.
            top_k (int): The number of top probable tokens to display.
            num_steps (int): The number of tokens to generate and visualize.
        """
        self.model_name: str = model_name
        self.top_k: int = top_k
        self.num_steps: int = num_steps
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        """
        Loads the pre-trained model and tokenizer.
        """
        try:
            logger.info(f"Loading model '{self.model_name}'...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            ).to(self.device)
            logger.info("Model loaded successfully.")

            logger.info(f"Loading tokenizer for model '{self.model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise

    def get_token_distribution(self, input_ids: torch.Tensor) -> Tuple[List[str], List[float]]:
        """
        Retrieves the top k token probabilities and their corresponding tokens for the given input IDs.

        Args:
            input_ids (torch.Tensor): The input tensor of token IDs.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing a list of token strings and their probabilities.
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer not loaded.")
            raise ValueError("Model or tokenizer not loaded.")

        try:
            logger.debug(f"Processing input IDs: {input_ids}")
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]  # Get logits for the last token
                probabilities = torch.softmax(logits, dim=-1).squeeze()

            top_probs, top_ids = torch.topk(probabilities, self.top_k)
            tokens = [
                self.tokenizer.decode([token_id]).strip() or "<UNK>" for token_id in top_ids.tolist()
            ]
            probs = top_probs.cpu().tolist()

            logger.debug(f"Top {self.top_k} tokens retrieved successfully.")
            return tokens, probs
        except Exception as e:
            logger.error(f"Error retrieving token distribution: {e}")
            raise

    def generate_tokens(self, prompt: str) -> List[str]:
        """
        Generates a list of tokens based on the provided prompt.

        Args:
            prompt (str): The initial text prompt to start token generation.

        Returns:
            List[str]: A list of generated tokens.
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer not loaded.")
            raise ValueError("Model or tokenizer not loaded.")

        try:
            logger.info("Starting token generation...")
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            generated_tokens: List[str] = []

            for step in range(self.num_steps):
                logger.debug(f"Generation step {step + 1}/{self.num_steps}")
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits = outputs.logits[:, -1, :]
                    probabilities = torch.softmax(logits, dim=-1).squeeze()

                top_probs, top_ids = torch.topk(probabilities, self.top_k)
                top_probs = top_probs.cpu().numpy()
                top_ids = top_ids.cpu().numpy()
                next_token_id = int(torch.multinomial(torch.tensor(top_probs), num_samples=1))
                next_token_id = top_ids[next_token_id]
                generated_tokens.append(self.tokenizer.decode([next_token_id]).strip() or "<UNK>")
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(self.device)], dim=1)

            logger.info("Token generation completed.")
            return generated_tokens
        except Exception as e:
            logger.error(f"Error during token generation: {e}")
            raise

    def visualize_generation(
        self, prompt: str, title: Optional[str] = None
    ) -> None:
        """
        Generates tokens based on the prompt and visualizes their probability distributions dynamically.

        Args:
            prompt (str): The initial text prompt to start token generation.
            title (Optional[str]): The title of the plot. If None, a default title is used.
        """
        try:
            tokens = self.generate_tokens(prompt)
            logger.info(f"Generated tokens: {tokens}")

            frames: List[Dict[str, Any]] = []
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            for step, token in enumerate(tokens, 1):
                tokens_frame, probs_frame = self.get_token_distribution(input_ids)
                token_texts = [t if t.strip() else "<UNK>" for t in tokens_frame]
                frame_title = f"Step {step}: Token '{token}' Generated"

                bar = go.Bar(
                    x=token_texts,
                    y=probs_frame,
                    marker=dict(
                        color=[
                            "rgba(255, 99, 71, 0.7)" if t == token else "rgba(55, 128, 191, 0.7)"
                            for t in tokens_frame
                        ]
                    ),
                    name="Probability",
                )

                frames.append(go.Frame(data=[bar], layout=go.Layout(title_text=frame_title)))

                # Update input_ids with the new token
                next_token_id = self.tokenizer.encode(token, add_special_tokens=False)
                if not next_token_id:
                    logger.warning(f"Token '{token}' could not be encoded. Skipping update.")
                    continue
                input_ids = torch.cat([input_ids, torch.tensor([next_token_id]).to(self.device)], dim=1)

            initial_tokens, initial_probs = self.get_token_distribution(input_ids)
            initial_token_texts = [t if t.strip() else "<UNK>" for t in initial_tokens]
            initial_bar = go.Bar(
                x=initial_token_texts,
                y=initial_probs,
                marker=dict(
                    color=[
                        "rgba(255, 99, 71, 0.7)" if t == tokens[0] else "rgba(55, 128, 191, 0.7)"
                        for t in initial_tokens
                    ]
                ),
                name="Probability",
            )

            plot_title = title or "Dynamic Token Probability Distribution"
            fig = go.Figure(
                data=[initial_bar],
                layout=go.Layout(
                    title=plot_title,
                    xaxis_title="Tokens",
                    yaxis_title="Probability",
                    updatemenus=[
                        dict(
                            type="buttons",
                            buttons=[
                                dict(
                                    label="Play",
                                    method="animate",
                                    args=[
                                        None,
                                        {
                                            "frame": {"duration": 500, "redraw": True},
                                            "fromcurrent": True,
                                            "transition": {"duration": 300},
                                        },
                                    ],
                                )
                            ],
                            showactive=False,
                            y=1.15,
                            x=1.05,
                        )
                    ],
                    showlegend=False,
                    template="plotly_white",
                ),
                frames=frames,
            )

            fig.update_layout(
                title=plot_title,
                xaxis_title="Tokens",
                yaxis_title="Probability",
                legend_title="Legend",
                bargap=0.2,
                template="plotly_white",
            )

            fig.show()
            logger.info("Dynamic visualization created successfully.")
        except Exception as e:
            logger.error(f"Error during visualization: {e}")
            raise


# def main() -> None:
#     """
#     Main function to execute the token visualization.
#     """
#     try:
#         # Configuration parameters
#         MODEL_NAME = "gpt2"       # Pre-trained model name
#         TOP_K = 100                # Number of top probable tokens to display
#         NUM_STEPS = 300            # Number of tokens to generate and visualize
#         PROMPT = "Once upon a time"  # Initial text prompt
#         TITLE = "Dynamic Token Probability Distribution"

#         # Initialize the visualizer with desired parameters
#         visualizer = TokenVisualizer(model_name=MODEL_NAME, top_k=TOP_K, num_steps=NUM_STEPS)

#         # Visualize the token distribution dynamically
#         visualizer.visualize_generation(prompt=PROMPT, title=TITLE)
#     except Exception as e:
#         logger.error(f"An error occurred in the main execution: {e}")


# if __name__ == "__main__":
#     main()