import os
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from timm import create_model
from tqdm import tqdm
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


class ImageClassificationFramework:
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the Image Classification Framework.

        Args:
            data_dir (str): Path to the dataset directory.
            batch_size (int): Batch size for data loading.
            num_workers (int): Number of workers for data loading.
            device (Optional[str]): Device to use for computations.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = (
            torch.device(device) if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.models: Dict[str, nn.Module] = {}
        self.results: Dict[str, Dict[str, List[float]]] = {}
        self.setup_logging()

    def setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare the dataset.

        Returns:
            Tuple[DataLoader, DataLoader]: Train and test data loaders.
        """
        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=transform)
            test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'test'), transform=transform)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )

            logging.info(f"Loaded dataset with {len(train_dataset)} training and {len(test_dataset)} testing images.")
            return train_loader, test_loader
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise

    def load_models(self) -> None:
        """Load all available models from timm."""
        try:
            model_names = timm.list_models(pretrained=True)
            for model_name in model_names:
                model = create_model(
                    model_name,
                    pretrained=True,
                    num_classes=len(self.train_loader.dataset.classes)
                )
                model = model.to(self.device)
                self.models[model_name] = model
            logging.info(f"Loaded {len(self.models)} models from timm.")
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise

    def train_model(self, model_name: str, epochs: int = 5) -> None:
        """
        Train a specific model.

        Args:
            model_name (str): Name of the model to train.
            epochs (int): Number of training epochs.
        """
        model = self.models[model_name]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        train_losses, test_losses, accuracies = [], [], []

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)

            test_loss, accuracy = self.evaluate_model(model)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            logging.info(f"{model_name} - Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

        self.results[model_name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies
        }

        self.save_model(model_name, model)

    def evaluate_model(self, model: nn.Module) -> Tuple[float, float]:
        """
        Evaluate a model on the test set.

        Args:
            model (nn.Module): The model to evaluate.

        Returns:
            Tuple[float, float]: Test loss and accuracy.
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(self.test_loader)
        accuracy = correct / total
        return test_loss, accuracy

    def save_model(self, model_name: str, model: nn.Module) -> None:
        """
        Save a trained model.

        Args:
            model_name (str): Name of the model.
            model (nn.Module): The model to save.
        """
        try:
            save_dir = os.path.join('saved_models', model_name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}.pth"))
            logging.info(f"Saved model weights for {model_name}")
        except Exception as e:
            logging.error(f"Error saving model {model_name}: {str(e)}")

    def plot_results(self) -> None:
        """Generate and save performance plots."""
        try:
            fig = make_subplots(rows=3, cols=1, subplot_titles=('Training Loss', 'Test Loss', 'Accuracy'))

            for model_name, result in self.results.items():
                fig.add_trace(go.Scatter(y=result['train_losses'], name=f"{model_name} - Train"), row=1, col=1)
                fig.add_trace(go.Scatter(y=result['test_losses'], name=f"{model_name} - Test"), row=2, col=1)
                fig.add_trace(go.Scatter(y=result['accuracies'], name=f"{model_name} - Accuracy"), row=3, col=1)

            fig.update_layout(height=900, width=800, title_text="Model Performance Comparison")
            plot(fig, filename='model_comparison.html', auto_open=False)
            logging.info("Saved performance plots to model_comparison.html")
        except Exception as e:
            logging.error(f"Error plotting results: {str(e)}")

    def run(self) -> None:
        """Execute the full training and evaluation pipeline."""
        try:
            self.train_loader, self.test_loader = self.load_dataset()
            self.load_models()

            with ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as executor:
                futures = [executor.submit(self.train_model, model_name) for model_name in self.models.keys()]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Error in model training: {str(e)}")

            self.plot_results()
        except Exception as e:
            logging.error(f"Error in running the framework: {str(e)}")


def main(data_dir: str) -> None:
    """
    Main function to run the Image Classification Framework.

    Args:
        data_dir (str): Path to the dataset directory.
    """
    try:
        framework = ImageClassificationFramework(data_dir=data_dir)
        framework.run()
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Classification Framework")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    args = parser.parse_args()

    main(args.data_dir)