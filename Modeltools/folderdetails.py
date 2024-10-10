from pathlib import Path
from typing import Union
from rich.console import Console
from rich.tree import Tree
from rich import print
import os


def create_tree_structure(
    directory: Union[str, Path], parent_tree: Tree = None
) -> Tree:
    """
    Recursively create a rich Tree structure for the given directory.

    Args:
        directory (Union[str, Path]): The directory path to visualize.
        parent_tree (Tree, optional): A parent tree to append directories/files to.

    Returns:
        Tree: A rich tree object representing the directory structure.
    """
    directory_path = Path(directory)

    if not directory_path.is_dir():
        raise ValueError(f"Invalid directory path: {directory_path}")

    if parent_tree is None:
        parent_tree = Tree(f"[bold blue]{directory_path.name}[/]")

    for path in sorted(directory_path.iterdir(), key=lambda p: (p.is_file(), p.name)):
        if path.is_dir():
            branch = parent_tree.add(f"[bold blue]{path.name}/[/]")
            create_tree_structure(path, branch)
        else:
            parent_tree.add(f"{path.name}")

    return parent_tree


def visualize_directory_structure(directory: Union[str, Path]) -> None:
    """
    Visualize the directory structure using the rich Tree.

    Args:
        directory (Union[str, Path]): The directory path to visualize.

    Raises:
        ValueError: If the provided directory path does not exist or is not a directory.
    """
    try:
        directory_path = Path(directory).resolve(strict=True)
        if not directory_path.is_dir():
            raise ValueError(f"The path {directory_path} is not a valid directory.")

        console = Console()
        tree = create_tree_structure(directory_path)
        console.print(tree)

    except FileNotFoundError as fnf_error:
        print(f"[bold red]Error:[/] {fnf_error}")
    except ValueError as val_error:
        print(f"[bold red]Error:[/] {val_error}")
    except Exception as generic_error:
        print(f"[bold red]An unexpected error occurred:[/] {generic_error}")

