import yaml
from typing import Any, Dict


class Config:
    def __init__(self, config_data: Dict[str, Any]) -> None:
        self.dataconfig = config_data.get('datasetconfig')
        self.modelconfig = config_data.get('modelconfig')
        self.promptconfig = config_data.get("promptconfig")
     
    
    @classmethod
    def from_yaml(cls, yaml_file: str, config_name: str) -> "Config":
        with open(yaml_file, 'r', encoding="UTF-8") as file:
            all_configs = yaml.safe_load(file)
            config_data = all_configs.get(config_name)
        return cls(config_data)
    
    def __str__(self) -> str:
        return f"Config(dataset={self.dataconfig}, model={self.modelconfig} prompttemplate={self.promptconfig} )"


def load_config(yml_path: str, user_values: Dict[str, Any]) -> Dict[str, Any]:
    with open(yml_path, 'r') as file:
        config = yaml.safe_load(file)

    def load_value(key: str, user_value: Any, default_value: Any) -> Any:
        return user_value if user_value is not None else default_value

    # Load the configuration with defaults or user-provided values
    for key, value in config.items():
        if isinstance(value, dict):
            default_value = value.get('default', None)
            user_value = user_values.get(key, None)
            config[key] = load_value(key, user_value, default_value)

    return config




# # load_config.py
# if __name__ == "__main__":
#     config = Config.from_yaml(r'E:\LLMS\Fine-tuning\LlmsComponents\Fine_tuning\congfig.yml', 'config1')
#     print(config.dataconfig,config.modelconfig)
#     print(config.dataconfig["DataConfig"]["path"])
#     # # Example usage:
#     user_values = {'do_train': True, 'per_device_train_batch_size': 16}
#     config = load_config('E:/LLMS/Fine-tuning/LlmsComponents/Fine_tuning/training.yml', user_values)
#     for key , valve in config.items():
#         print(f"key:{key}  valve:  {valve}")
