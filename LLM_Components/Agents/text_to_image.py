import os
import time
from typing import Optional, Tuple, List

import torch
import numpy as np
from PIL import Image
from IPython import display as IPdisplay
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from transformers import logging

logging.set_verbosity_error()

# Set device configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def load_model(model_name_or_path: str, scheduler: LMSDiscreteScheduler) -> StableDiffusionPipeline:
    """Load the Stable Diffusion model with specified scheduler."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name_or_path,
        scheduler=scheduler,
        torch_dtype=torch.float32,
        cache_dir="./model_cache",
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_model_cpu_offload()
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_xformers_memory_efficient_attention()
    return pipe


def display_images(images: List[Image.Image], save_path: str) -> IPdisplay.Image:
    """Display and save generated images as a GIF."""
    try:
        filename = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        images[0].save(
            os.path.join(save_path, f"{filename}.gif"),
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
        )
    except Exception as e:
        print(f"Error while saving images: {e}")
    return IPdisplay.Image(os.path.join(save_path, f"{filename}.gif"))


def tokenize_prompts(pipe: StableDiffusionPipeline, prompt: str, negative_prompt: Optional[str] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize and encode the prompts."""
    prompt_tokens = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds = pipe.text_encoder(prompt_tokens.input_ids.to(device))[0]

    if negative_prompt is None:
        negative_prompt = ""

    negative_prompt_tokens = pipe.tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    negative_prompt_embeds = pipe.text_encoder(negative_prompt_tokens.input_ids.to(device))[0]

    return prompt_embeds, negative_prompt_embeds


def generate_latents(pipe: StableDiffusionPipeline, height: int, width: int,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Generate initial latent vectors."""
    return torch.randn(
        (1, pipe.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    ).to(device)


def interpolate_embeddings(prompt_embeds: torch.Tensor, negative_prompt_embeds: torch.Tensor,
                           num_interpolation_steps: int, step_size: float) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Interpolate between embeddings."""
    walked_embeddings = []
    for i in range(num_interpolation_steps):
        walked_embeddings.append(
            (prompt_embeds + step_size * i, negative_prompt_embeds + step_size * i)
        )
    return walked_embeddings


def generate_images(pipe: StableDiffusionPipeline, walked_embeddings: List[Tuple[torch.Tensor, torch.Tensor]],
                    latents: torch.Tensor, height: int, width: int, num_inference_steps: int,
                    guidance_scale: float, generator: Optional[torch.Generator]) -> List[Image.Image]:
    """Generate images from embeddings."""
    images = []
    for latent in tqdm(walked_embeddings):
        generated = pipe(
            height=height,
            width=width,
            num_images_per_prompt=1,
            prompt_embeds=latent[0],
            negative_prompt_embeds=latent[1],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=latents,
        ).images
        images.append(generated[0])
    return images


def main() -> None:
    """Main execution logic."""
    save_path = "./output"
    os.makedirs(save_path, exist_ok=True)

    prompt = ("Epic shot of Sweden, ultra detailed lake with a reindeer, nostalgic vintage, ultra cozy and inviting, "
              "wonderful light atmosphere, fairy, little photorealistic, digital painting, sharp focus, ultra cozy and inviting, "
              "wish to be there. very detailed, arty, should rank high on youtube for a dream trip.")
    negative_prompt = ("poorly drawn, cartoon, 2d, disfigured, bad art, deformed, poorly drawn, extra limbs, close up, b&w, "
                       "weird colors, blurry")
    seed = 2
    guidance_scale = 8
    num_inference_steps = 15
    num_interpolation_steps = 30
    height = 512
    width = 512
    step_size = 0.001

    generator = torch.manual_seed(seed) if seed is not None else None

    # Load the model
    model_name_or_path = "runwayml/stable-diffusion-v1-5"
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    pipe = load_model(model_name_or_path, scheduler)

    prompt_embeds, negative_prompt_embeds = tokenize_prompts(pipe, prompt, negative_prompt)
    latents = generate_latents(pipe, height, width, generator)
    walked_embeddings = interpolate_embeddings(prompt_embeds, negative_prompt_embeds, num_interpolation_steps, step_size)
    images = generate_images(pipe, walked_embeddings, latents, height, width, num_inference_steps, guidance_scale, generator)
    display_images(images, save_path)


if __name__ == "__main__":
    main()
    
    
# import os
# import time
# from typing import List, Optional, Tuple

# import numpy as np
# import torch
# from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
# from IPython import display as IPdisplay
# from PIL import Image
# from tqdm.auto import tqdm
# from transformers import logging

# logging.set_verbosity_error()

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


# def load_model(model_name_or_path: str, scheduler: LMSDiscreteScheduler) -> StableDiffusionPipeline:
#     """Load the Stable Diffusion model."""
#     pipe = StableDiffusionPipeline.from_pretrained(
#         model_name_or_path,
#         scheduler=scheduler,
#         torch_dtype=torch.float32,
#         cache_dir="./model_cache",
#     ).to(DEVICE)
#     pipe.set_progress_bar_config(disable=True)
#     pipe.enable_model_cpu_offload()
#     pipe.unet.to(memory_format=torch.channels_last)
#     pipe.enable_vae_slicing()
#     pipe.enable_vae_tiling()
#     pipe.enable_xformers_memory_efficient_attention()
#     return pipe


# def display_images(images: List[Image.Image], save_path: str) -> IPdisplay.Image:
#     """Display and save images as GIF."""
#     try:
#         filename = time.strftime("%H-%M-%S", time.localtime())
#         filepath = os.path.join(save_path, f"{filename}.gif")
#         images[0].save(
#             filepath,
#             save_all=True,
#             append_images=images[1:],
#             duration=100,
#             loop=0,
#         )
#         return IPdisplay.Image(filepath)
#     except Exception as e:
#         print(f"Error saving GIF: {e}")
#         return None


# def tokenize_prompts(
#     pipe: StableDiffusionPipeline, prompt: str, negative_prompt: Optional[str] = None
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Tokenize prompts."""
#     prompt_tokens = pipe.tokenizer(
#         prompt,
#         padding="max_length",
#         max_length=pipe.tokenizer.model_max_length,
#         truncation=True,
#         return_tensors="pt",
#     )
#     prompt_embeds = pipe.text_encoder(prompt_tokens.input_ids.to(DEVICE))[0]

#     negative_prompt = negative_prompt or ""
#     negative_prompt_tokens = pipe.tokenizer(
#         negative_prompt,
#         padding="max_length",
#         max_length=pipe.tokenizer.model_max_length,
#         truncation=True,
#         return_tensors="pt",
#     )
#     negative_prompt_embeds = pipe.text_encoder(negative_prompt_tokens.input_ids.to(DEVICE))[0]

#     return prompt_embeds, negative_prompt_embeds


# def generate_latents(
#     pipe: StableDiffusionPipeline, height: int, width: int, generator: Optional[torch.Generator] = None
# ) -> torch.Tensor:
#     """Generate initial latent vectors."""
#     return torch.randn(
#         (1, pipe.unet.config.in_channels, height // 8, width // 8),
#         generator=generator,
#         device=DEVICE,
#     )


# def interpolate_embeddings(
#     prompt_embeds: torch.Tensor,
#     negative_prompt_embeds: torch.Tensor,
#     num_interpolation_steps: int,
#     step_size: float,
# ) -> List[List[torch.Tensor]]:
#     """Interpolate between embeddings."""
#     return [
#         [prompt_embeds + step_size * i, negative_prompt_embeds + step_size * i]
#         for i in range(num_interpolation_steps)
#     ]


# def generate_images(
#     pipe: StableDiffusionPipeline,
#     walked_embeddings: List[List[torch.Tensor]],
#     latents: torch.Tensor,
#     height: int,
#     width: int,
#     num_inference_steps: int,
#     guidance_scale: float,
#     generator: Optional[torch.Generator],
# ) -> List[Image.Image]:
#     """Generate images from embeddings."""
#     images = []
#     for latent in tqdm(walked_embeddings):
#         image = pipe(
#             height=height,
#             width=width,
#             num_images_per_prompt=1,
#             prompt_embeds=latent[0],
#             negative_prompt_embeds=latent[1],
#             num_inference_steps=num_inference_steps,
#             guidance_scale=guidance_scale,
#             generator=generator,
#             latents=latents,
#         ).images[0]
#         images.append(image)
#     return images


# def main():
#     save_path = "./output"
#     os.makedirs(save_path, exist_ok=True)

#     prompt = "Epic shot of Sweden, ultra detailed lake with a reindeer, nostalgic vintage, ultra cozy and inviting, wonderful light atmosphere, fairy, little photorealistic, digital painting, sharp focus, ultra cozy and inviting, wish to be there. very detailed, arty, should rank high on youtube for a dream trip."
#     negative_prompt = "poorly drawn,cartoon, 2d, disfigured, bad art, deformed, poorly drawn, extra limbs, close up, b&w, weird colors, blurry"
#     seed = 2
#     guidance_scale = 8
#     num_inference_steps = 15
#     num_interpolation_steps = 30
#     height = 512
#     width = 512
#     step_size = 0.001

#     generator = torch.manual_seed(seed) if seed is not None else None

#     model_name_or_path = "runwayml/stable-diffusion-v1-5"
#     scheduler = LMSDiscreteScheduler(
#         beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
#     )
#     pipe = load_model(model_name_or_path, scheduler)

#     prompt_embeds, negative_prompt_embeds = tokenize_prompts(pipe, prompt, negative_prompt)
#     latents = generate_latents(pipe, height, width, generator)
#     walked_embeddings = interpolate_embeddings(
#         prompt_embeds, negative_prompt_embeds, num_interpolation_steps, step_size
#     )
#     images = generate_images(
#         pipe, walked_embeddings, latents, height, width, num_inference_steps, guidance_scale, generator
#     )
#     display_images(images, save_path)


# if __name__ == "__main__":
#     main()