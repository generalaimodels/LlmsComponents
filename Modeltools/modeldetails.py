import torch
from torch import nn
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel

def model_summary(model: nn.Module) -> None:
    """
    Generate a detailed summary of a PyTorch model, including the number of total,
    trainable, and non-trainable parameters. Also, visually display the model architecture
    using a tree structure.

    Args:
        model (nn.Module): The PyTorch model to summarize.

    Raises:
        TypeError: If the input is not a PyTorch nn.Module.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("Input must be a PyTorch nn.Module")

    console = Console()

    # Compute parameters' stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Create a caption for the model
    model_caption = f"[bold cyan]{model.__class__.__name__}[/bold cyan]"
    console.print(Panel(model_caption, expand=False))

    # Create a summary table for the total parameters
    total_params_table = Table(
        title="Total Parameters Summary", 
        show_header=True, 
        header_style="bold magenta"
    )
    total_params_table.add_column("Category", style="cyan")
    total_params_table.add_column("Count", style="green", justify="right")

    total_params_table.add_row("Total Parameters", f"{total_params:,}")
    total_params_table.add_row("Trainable Parameters", f"{trainable_params:,}")
    total_params_table.add_row("Non-trainable Parameters", f"{non_trainable_params:,}")

    console.print(total_params_table)
    console.print(Panel("[italic]Summary of model's total parameters[/italic]", expand=False))

    # Create a detailed table for the parameters
    detailed_table = Table(
        title="Detailed Parameter Summary", 
        show_lines=True
    )
    detailed_table.add_column("Parameter", style="cyan", no_wrap=True)
    detailed_table.add_column("Shape", style="magenta")
    detailed_table.add_column("Trainable", style="green")
    detailed_table.add_column("Num Parameters", style="yellow", justify="right")

    def add_module_to_table(module: nn.Module, prefix: Optional[str] = "") -> None:
        """
        Recursively add model parameters to the detailed table.

        Args:
            module (nn.Module): A PyTorch module
            prefix (Optional[str]): Prefix for parameter names, used for recursion.
        """
        for name, param in module.named_parameters(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            shape = tuple(param.shape)
            trainable = param.requires_grad
            num_params = param.numel()
            
            detailed_table.add_row(
                full_name,
                str(shape),
                "[green]✓[/green]" if trainable else "[red]✗[/red]",
                f"{num_params:,}"
            )

        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            add_module_to_table(child, child_prefix)

    add_module_to_table(model)
    console.print(detailed_table)
    console.print(Panel("[italic]Detailed breakdown of model's parameters[/italic]", expand=False))

    # Create a rich Tree for the model architecture
    def build_tree(module: nn.Module, prefix: Optional[str] = "") -> Tree:
        """
        Recursively build a tree structure for the model architecture.

        Args:
            module (nn.Module): The module to represent in the tree.
            prefix (Optional[str]): The prefix for the current module.

        Returns:
            Tree: A rich Tree representing the module architecture.
        """
        root_name = prefix if prefix else module.__class__.__name__
        tree = Tree(f"[bold]{root_name}[/bold]")

        for name, child in module.named_children():
            child_tree = build_tree(child, prefix=name)
            tree.add(child_tree)

        for name, param in module.named_parameters(recurse=False):
            param_info = f"{name}: {tuple(param.shape)} {'(Trainable)' if param.requires_grad else '(Non-trainable)'}"
            tree.add(f"[cyan]{param_info}[/cyan]")

        return tree

    architecture_tree = build_tree(model)
    console.print("\n[bold yellow]Model Architecture[/bold yellow]")
    console.print(architecture_tree)
    console.print(Panel("[italic]Visual representation of the model's architecture[/italic]", expand=False))




def model_summary_update(model: nn.Module, save_path: Optional[str] = "model_summary.html") -> None:
    """
    Generate a detailed summary of a PyTorch model, including the number of total,
    trainable, and non-trainable parameters. Also, visually display the model architecture
    using a tree structure.

    Args:
        model (nn.Module): The PyTorch model to summarize.
        save_path (Optional[str]): Path to save the summary as an HTML file.

    Raises:
        TypeError: If the input is not a PyTorch nn.Module.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("Input must be a PyTorch nn.Module")

    console = Console(record=True)  # Enable recording of console output

    # Compute parameters' stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Create a caption for the model
    model_caption = f"[bold cyan]{model.__class__.__name__}[/bold cyan]"
    console.print(Panel(model_caption, expand=False))

    # Create a summary table for the total parameters
    total_params_table = Table(
        title="Total Parameters Summary", 
        show_header=True, 
        header_style="bold magenta"
    )
    total_params_table.add_column("Category", style="cyan")
    total_params_table.add_column("Count", style="green", justify="right")

    total_params_table.add_row("Total Parameters", f"{total_params:,}")
    total_params_table.add_row("Trainable Parameters", f"{trainable_params:,}")
    total_params_table.add_row("Non-trainable Parameters", f"{non_trainable_params:,}")

    console.print(total_params_table)
    console.print(Panel("[italic]Summary of model's total parameters[/italic]", expand=False))

    # Create a detailed table for the parameters
    detailed_table = Table(
        title="Detailed Parameter Summary", 
        show_lines=True
    )
    detailed_table.add_column("Parameter", style="cyan", no_wrap=True)
    detailed_table.add_column("Shape", style="magenta")
    detailed_table.add_column("Trainable", style="green")
    detailed_table.add_column("Num Parameters", style="yellow", justify="right")

    def add_module_to_table(module: nn.Module, prefix: Optional[str] = "") -> None:
        """
        Recursively add model parameters to the detailed table.

        Args:
            module (nn.Module): A PyTorch module
            prefix (Optional[str]): Prefix for parameter names, used for recursion.
        """
        for name, param in module.named_parameters(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            shape = tuple(param.shape)
            trainable = param.requires_grad
            num_params = param.numel()
            
            detailed_table.add_row(
                full_name,
                str(shape),
                "[green]✓[/green]" if trainable else "[red]✗[/red]",
                f"{num_params:,}"
            )

        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            add_module_to_table(child, child_prefix)

    add_module_to_table(model)
    console.print(detailed_table)
    console.print(Panel("[italic]Detailed breakdown of model's parameters[/italic]", expand=False))

    # Create a rich Tree for the model architecture
    def build_tree(module: nn.Module, prefix: Optional[str] = "") -> Tree:
        """
        Recursively build a tree structure for the model architecture.

        Args:
            module (nn.Module): The module to represent in the tree.
            prefix (Optional[str]): The prefix for the current module.

        Returns:
            Tree: A rich Tree representing the module architecture.
        """
        root_name = prefix if prefix else module.__class__.__name__
        tree = Tree(f"[bold]{root_name}[/bold]")

        for name, child in module.named_children():
            child_tree = build_tree(child, prefix=name)
            tree.add(child_tree)

        for name, param in module.named_parameters(recurse=False):
            param_info = f"{name}: {tuple(param.shape)} {'(Trainable)' if param.requires_grad else '(Non-trainable)'}"
            tree.add(f"[cyan]{param_info}[/cyan]")

        return tree

    architecture_tree = build_tree(model)
    console.print("\n[bold yellow]Model Architecture[/bold yellow]")
    console.print(architecture_tree)
    console.print(Panel("[italic]Visual representation of the model's architecture[/italic]", expand=False))

    if save_path:
        html = console.export_html(clear=False)  # Removed code_format parameter
        with open(save_path, "w") as f:
            f.write(html)
        print(f"Summary saved as {save_path}")


