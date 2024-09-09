import os
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


class ObjectDetectionFramework:
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 4,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the Object Detection Framework.

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
                transforms.ToTensor(),
            ])

            train_dataset = torchvision.datasets.CocoDetection(
                root=os.path.join(self.data_dir, 'train'),
                annFile=os.path.join(self.data_dir, 'train', '_annotations.coco.json'),
                transform=transform
            )
            test_dataset = torchvision.datasets.CocoDetection(
                root=os.path.join(self.data_dir, 'test'),
                annFile=os.path.join(self.data_dir, 'test', '_annotations.coco.json'),
                transform=transform
            )

            def collate_fn(batch):
                return tuple(zip(*batch))

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_fn
            )

            logging.info(f"Loaded dataset with {len(train_dataset)} training and {len(test_dataset)} testing images.")
            return train_loader, test_loader
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise

    def load_models(self) -> None:
        """Load Faster R-CNN model."""
        try:
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            num_classes = 91  # COCO dataset has 90 classes + background
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            model = model.to(self.device)
            self.models['faster_rcnn'] = model
            logging.info("Loaded Faster R-CNN model.")
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise

    def train_model(self, model_name: str, epochs: int = 10) -> None:
        """
        Train a specific model.

        Args:
            model_name (str): Name of the model to train.
            epochs (int): Number of training epochs.
        """
        model = self.models[model_name]
        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

        train_losses, test_losses = [], []

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for images, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                train_loss += losses.item()

            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)

            test_loss = self.evaluate_model(model)
            test_losses.append(test_loss)

            logging.info(f"{model_name} - Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        self.results[model_name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
        }

        self.save_model(model_name, model)

    def evaluate_model(self, model: nn.Module) -> float:
        """
        Evaluate a model on the test set.

        Args:
            model (nn.Module): The model to evaluate.

        Returns:
            float: Test loss.
        """
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for images, targets in self.test_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                test_loss += losses.item()

        test_loss /= len(self.test_loader)
        return test_loss

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
            fig = make_subplots(rows=2, cols=1, subplot_titles=('Training Loss', 'Test Loss'))

            for model_name, result in self.results.items():
                fig.add_trace(go.Scatter(y=result['train_losses'], name=f"{model_name} - Train"), row=1, col=1)
                fig.add_trace(go.Scatter(y=result['test_losses'], name=f"{model_name} - Test"), row=2, col=1)

            fig.update_layout(height=600, width=800, title_text="Model Performance Comparison")
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
    Main function to run the Object Detection Framework.

    Args:
        data_dir (str): Path to the dataset directory.
    """
    try:
        framework = ObjectDetectionFramework(data_dir=data_dir)
        framework.run()
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Object Detection Framework")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    args = parser.parse_args()

    main(args.data_dir)