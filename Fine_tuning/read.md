```python
data_config:
  train_file: "path/to/train.csv"
  validation_file: "path/to/validation.csv"
  test_file: "path/to/test.csv"
  text_column: "text"
  label_column: "label"

model_config:
  model_name_or_path: "bert-base-uncased"
  num_labels: 2

prompt_template:
  template: "Classify the following text: {text}\nLabel:"

training_args:
  output_dir: "./results"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  warmup_steps: 500
  weight_decay: 0.01
  logging_dir: "./logs"
  logging_steps: 10
  evaluation_strategy: "steps"
  eval_steps: 500
  save_steps: 1000
  load_best_model_at_end: true

hub_repo_name: "your-username/your-model-name"

```