from accelerate import Accelerator


accelerator = Accelerator()

accelerator.print(f"Accelerator state from the current environment:\n{accelerator.state}")
if accelerator.fp8_recipe_handler is not None:
    accelerator.print(f"FP8 config:\n{accelerator.fp8_recipe_handler}")
accelerator.end_training()