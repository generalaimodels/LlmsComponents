from typing import Union, Tuple, Optional
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


class AdvancedTextImageInferencePipeline:
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        model_name_or_path: str = "microsoft/Florence-2-large"
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            cache_dir=cache_dir
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

    def load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image_source, str):
            if image_source.startswith(('http://', 'https://')):
                response = requests.get(image_source, stream=True)
                response.raise_for_status()
                return Image.open(response.raw)
            else:
                return Image.open(image_source)
        elif isinstance(image_source, Image.Image):
            return image_source
        else:
            raise ValueError("Invalid image source. Must be a URL, local path, or PIL Image.")

    def generate(
        self,
        image_source: Union[str, Image.Image],
        prompt: str,
        max_new_tokens: int = 1024,
        num_beams: int = 3,
        do_sample: bool = False
    ) -> Tuple[str, dict]:
        image = self.load_image(image_source)
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )

        return generated_text, parsed_answer

    def __call__(
        self,
        image_source: Union[str, Image.Image],
        prompt: str,
        max_new_tokens: int = 1024,
        num_beams: int = 3,
        do_sample: bool = False
    ) -> Tuple[str, dict]:
        return self.generate(image_source, prompt, max_new_tokens, num_beams, do_sample)


if __name__ == "__main__":
    pipeline = AdvancedTextImageInferencePipeline()
    
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    prompt = "<OD>"
    
    generated_text, parsed_answer = pipeline(url, prompt)
    
    print("Generated Text:")
    print(generated_text)
    print("\nParsed Answer:")
    print(parsed_answer)