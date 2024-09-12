import torch
from torch import nn
from typing import Dict, Any, Union
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.text import Text
from typing import NoReturn
from datasets import  DatasetDict
from rich.console import Console
from rich.table import Table
from rich import box

def model_summary(model: nn.Module) -> None:
    """
    Generate a detailed summary of a PyTorch model, including parameter statistics
    and a visual representation of the model architecture.

    Args:
        model (nn.Module): The PyTorch model to summarize.

    Raises:
        TypeError: If the input is not a PyTorch nn.Module.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("Input must be a PyTorch nn.Module")

    console = Console()

    # Compute parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Display model header
    model_caption = Text(model.__class__.__name__, style="bold cyan")
    console.print(Panel(model_caption, title="Model Summary", expand=False))

    # Display parameter summary table
    param_summary_table = Table(title="Parameter Statistics", show_header=True, header_style="bold magenta")
    param_summary_table.add_column("Category", style="cyan")
    param_summary_table.add_column("Count", style="green", justify="right")

    param_summary_table.add_row("Total Parameters", f"{total_params:,}")
    param_summary_table.add_row("Trainable Parameters", f"{trainable_params:,}")
    param_summary_table.add_row("Non-trainable Parameters", f"{non_trainable_params:,}")

    console.print(param_summary_table)
    console.print(Panel(Text("Overview of model's parameter distribution", style="italic"), expand=False))

    # Display detailed parameter table
    detailed_table = Table(title="Detailed Parameter Breakdown", show_lines=True)
    detailed_table.add_column("Parameter", style="cyan", no_wrap=True)
    detailed_table.add_column("Shape", style="magenta")
    detailed_table.add_column("Trainable", style="green")
    detailed_table.add_column("Num Parameters", style="yellow", justify="right")

    def add_module_to_table(module: nn.Module, prefix: str = "") -> None:
        """
        Recursively add module parameters to the detailed table.

        Args:
            module (nn.Module): A PyTorch module
            prefix (str): Prefix for parameter names, used for recursion
        """
        for name, param in module.named_parameters(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            shape = tuple(param.shape)
            trainable = param.requires_grad
            num_params = param.numel()
            
            detailed_table.add_row(
                full_name,
                str(shape),
                "✓" if trainable else "✗",
                f"{num_params:,}"
            )

        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            add_module_to_table(child, child_prefix)

    add_module_to_table(model)
    console.print(detailed_table)
    console.print(Panel(Text("Comprehensive breakdown of model's parameters", style="italic"), expand=False))

    # Create a rich Tree for the model architecture
    def build_architecture_tree(module: nn.Module, prefix: str = "") -> Tree:
        """
        Recursively build a tree structure representing the model architecture.

        Args:
            module (nn.Module): The module to represent in the tree
            prefix (str): The prefix for the current module

        Returns:
            Tree: A rich Tree representing the module architecture
        """
        root_name = prefix if prefix else module.__class__.__name__
        tree = Tree(Text(root_name, style="bold"))

        for name, child in module.named_children():
            child_tree = build_architecture_tree(child, prefix=name)
            tree.add(child_tree)

        for name, param in module.named_parameters(recurse=False):
            param_info = f"{name}: {tuple(param.shape)} ({'Trainable' if param.requires_grad else 'Non-trainable'})"
            tree.add(Text(param_info, style="cyan"))

        return tree

    architecture_tree = build_architecture_tree(model)
    console.print("\n")
    console.print(Text("Model Architecture", style="bold yellow"))
    console.print(architecture_tree)
    console.print(Panel(Text("Visual representation of the model's structure", style="italic"), expand=False))


def visualize_dataset(dataset: DatasetDict) -> NoReturn:
    """
    Visualize the structure of a HuggingFace DatasetDict using the rich library.

    Args:
        dataset (DatasetDict): The dataset to visualize.

    Returns:
        NoReturn: This function prints the visualization and does not return anything.
    """
    console = Console()
    
    # Create a table for dataset overview
    table = Table(title=" 🗃️   Dataset Overview", box=box.ROUNDED, title_justify="left")

    table.add_column("Split", justify="center", style="cyan", no_wrap=True)
    table.add_column("Features", justify="center", style="magenta")
    table.add_column("Number of Rows", justify="center", style="green")

    # Loop through the dataset dict to populate the table
    for split, data in dataset.items():
        features = ', '.join(data.column_names)
        num_rows = str(data.num_rows)
        
        table.add_row(split, features, num_rows)

    console.print(table)



