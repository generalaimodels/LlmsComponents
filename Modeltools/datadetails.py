from typing import NoReturn
from datasets import  DatasetDict
from rich.console import Console
from rich.table import Table
from rich import box

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



