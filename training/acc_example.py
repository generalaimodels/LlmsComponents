import argparse
from typing import Dict, Any, Tuple
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_linear_schedule_with_warmup
from datasets import Dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
import evaluate
from typing import Dict, Any
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
class AcceleratedTrainer:
    def __init__(self, 
                 model: PreTrainedModel, 
                 tokenizer: PreTrainedTokenizerBase, 
                 optimizer: Optimizer, 
                 train_dataset: Dataset, 
                 eval_dataset: Dataset, 
                 metric: Any,
                 config: Dict[str, Any],
                 accelerator: Accelerator):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metric = metric
        self.config = config
        self.accelerator = accelerator

    def prepare_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        try:
            train_dataloader = DataLoader(
                self.train_dataset,
                shuffle=True,
                collate_fn=self.collate_fn,
                batch_size=self.config['batch_size'],
                drop_last=True
            )
            eval_dataloader = DataLoader(
                self.eval_dataset,
                shuffle=False,
                collate_fn=self.collate_fn,
                batch_size=self.config['eval_batch_size'],
                drop_last=(self.accelerator.mixed_precision == "fp8")
            )
            return train_dataloader, eval_dataloader
        except Exception as e:
            self.accelerator.print(f"Error in preparing dataloaders: {str(e)}")
            raise

    def collate_fn(self, examples: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        try:
            max_length = None if self.accelerator.distributed_type != "TPU" else 128
            pad_to_multiple_of = 16 if self.accelerator.mixed_precision == "fp8" else 8 if self.accelerator.mixed_precision != "no" else None
            
            return self.tokenizer.pad(
                examples,
                padding="longest",
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt"
            )
        except Exception as e:
            self.accelerator.print(f"Error in collate function: {str(e)}")
            raise

    def train(self):
        try:
            train_dataloader, eval_dataloader = self.prepare_dataloaders()
            
            num_update_steps_per_epoch = len(train_dataloader) // self.config['gradient_accumulation_steps']
            max_train_steps = self.config['num_epochs'] * num_update_steps_per_epoch
            
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.config['num_warmup_steps'],
                num_training_steps=max_train_steps,
            )

            self.model, self.optimizer, train_dataloader, eval_dataloader, lr_scheduler = \
                self.accelerator.prepare(self.model, self.optimizer, train_dataloader, eval_dataloader, lr_scheduler)

            total_batch_size = self.config['batch_size'] * self.accelerator.num_processes * self.config['gradient_accumulation_steps']
            
            self.accelerator.print("***** Running training *****")
            self.accelerator.print(f"  Num examples = {len(self.train_dataset)}")
            self.accelerator.print(f"  Num Epochs = {self.config['num_epochs']}")
            self.accelerator.print(f"  Instantaneous batch size per device = {self.config['batch_size']}")
            self.accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            self.accelerator.print(f"  Gradient Accumulation steps = {self.config['gradient_accumulation_steps']}")
            self.accelerator.print(f"  Total optimization steps = {max_train_steps}")
            
            progress_bar = self.accelerator.init_trackers("training")
            completed_steps = 0
            
            for epoch in range(self.config['num_epochs']):
                self.model.train()
                for step, batch in enumerate(train_dataloader):
                    with self.accelerator.accumulate(self.model):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        self.accelerator.backward(loss)
                        self.optimizer.step()
                        lr_scheduler.step()
                        self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)
                        completed_steps += 1

                    if completed_steps >= max_train_steps:
                        break

                self.evaluate(eval_dataloader)

            self.accelerator.end_training()

        except Exception as e:
            self.accelerator.print(f"Error during training: {str(e)}")
            raise

    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_predictions = []
        all_references = []
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(total=len(eval_dataloader), desc="Evaluation", disable=not self.accelerator.is_local_main_process)

        try:
            for batch in eval_dataloader:
                with torch.no_grad():
                    outputs = self.model(**batch)
                
                loss = outputs.loss
                total_loss += loss.detach().float()
                num_batches += 1

                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = self.accelerator.gather_for_metrics((predictions, batch["labels"]))
                
                all_predictions.extend(predictions.cpu().numpy())
                all_references.extend(references.cpu().numpy())
                
                progress_bar.update(1)

            progress_bar.close()

            # Compute metrics
            eval_metric = self.metric.compute(predictions=all_predictions, references=all_references)
            
            # Add average loss to metrics
            eval_metric['average_loss'] = total_loss.item() / num_batches

            # Log results
            self.accelerator.print("\nEvaluation Results:")
            for key, value in eval_metric.items():
                self.accelerator.print(f"{key}: {value}")

            return eval_metric

        except Exception as e:
            self.accelerator.print(f"Error during evaluation: {str(e)}")
            raise
        finally:
            progress_bar.close()

# def main():
#     parser = argparse.ArgumentParser(description="Advanced Accelerated Training Script")
#     parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16", "fp8"])
#     parser.add_argument("--cpu", action="store_true")
#     args = parser.parse_args()

#     config = {
#         "lr": 2e-5,
#         "num_epochs": 3,
#         "seed": 42,
#         "batch_size": 16,
#         "eval_batch_size": 32,
#         "gradient_accumulation_steps": 1,
#         "num_warmup_steps": 100,
#     }

#     accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)
#     set_seed(config['seed'])

#     # Here you would initialize your model, tokenizer, datasets, and metric
#     # model = ...
#     # tokenizer = ...
#     # train_dataset = ...
#     # eval_dataset = ...
#     # metric = ...

#     trainer = AcceleratedTrainer(model, tokenizer, optimizer, train_dataset, eval_dataset, metric, config, accelerator)
#     trainer.train()

# if __name__ == "__main__":
#     main()