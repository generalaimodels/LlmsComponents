from typing import List, Optional, Tuple, Dict, Any, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
)
import torch
from dataclasses import dataclass
from enum import Enum
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class GenerationType(Enum):
    """Enum representing different text generation strategies."""
    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"
    SAMPLING = "sampling"
    CONTRASTIVE_SEARCH = "contrastive_search"
    NUCLEUS_SAMPLING = "nucleus_sampling"
    TOP_K_SAMPLING = "top_k_sampling"
    DIVERSE_BEAM_SEARCH = "diverse_beam_search"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"

@dataclass
class GenerationParameters:
    """Dataclass holding parameters for text generation."""
    max_length: int = 50
    num_beams: int = 5
    temperature: float = 1.0
    top_k: Optional[int] = 100
    top_p: Optional[float] = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1

class TextGenerationPipeline:
    """
    Pipeline for advanced text generation using Hugging Face Transformers.
    """

    def __init__(self, model_name: str):
        """
        Initializes the pipeline with a pre-trained language model.

        Args:
            model_name: Name of the pre-trained model from Hugging Face Model Hub.
        """
        try:
            self.config = AutoConfig.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def generate_text(
        self,
        text: List[str],
        generation_type: GenerationType = GenerationType.GREEDY,
        generation_params: Optional[GenerationParameters] = None,
    ) -> Tuple[List[str], Optional[List[torch.Tensor]]]:
        if generation_params is None:
            generation_params = GenerationParameters()

        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

            # Create a base GenerationConfig
            gen_config = GenerationConfig(
                max_length=generation_params.max_length,
                temperature=generation_params.temperature,
                num_beams=generation_params.num_beams,
                do_sample=generation_params.do_sample,
                top_k=generation_params.top_k,
                top_p=generation_params.top_p,
                repetition_penalty=generation_params.repetition_penalty,
                num_return_sequences=generation_params.num_return_sequences,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Adjust generation config based on generation type
            if generation_type == GenerationType.GREEDY:
                gen_config.num_beams = 1
                gen_config.do_sample = False
            elif generation_type == GenerationType.BEAM_SEARCH:
                gen_config.num_return_sequences = gen_config.num_beams
            elif generation_type == GenerationType.SAMPLING:
                gen_config.do_sample = True
            elif generation_type == GenerationType.CONTRASTIVE_SEARCH:
                gen_config.penalty_alpha = 0.6
                gen_config.top_k = 4
            elif generation_type == GenerationType.NUCLEUS_SAMPLING:
                gen_config.do_sample = True
                gen_config.top_p = 0.95
            elif generation_type == GenerationType.TOP_K_SAMPLING:
                gen_config.do_sample = True
                gen_config.top_k = 50
            elif generation_type == GenerationType.DIVERSE_BEAM_SEARCH:
                gen_config.num_beam_groups = 3
                gen_config.diversity_penalty = 0.5
            elif generation_type == GenerationType.CONSTRAINED_BEAM_SEARCH:
                force_words = ["amazing", "wonderful"]
                force_words_ids = [self.tokenizer(word, add_special_tokens=False).input_ids for word in force_words]
                gen_config.force_words_ids = force_words_ids
            else:
                raise ValueError(f"Invalid generation type: {generation_type}")

            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=gen_config
            )

            generated_text = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            scores = outputs.scores if hasattr(outputs, "scores") else None

            return generated_text, scores

        except Exception as e:
            raise RuntimeError(f"Error during text generation: {e}")

    def visualize_generation(
        self, 
        scores: Optional[List[torch.Tensor]], 
        input_text: str,
        generated_text: str
    ) -> None:
        """
        Visualizes the text generation process using Plotly.
        """
        if not scores:
            raise ValueError("Scores are required for visualization")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Token Probabilities Over Time",
                "Top 10 Token Probabilities (Final Step)",
                "Temperature Effect on Token Probabilities",
                "Text Generation Process"
            )
        )

        # Token Probabilities Over Time
        token_probs = torch.stack([torch.nn.functional.softmax(score, dim=-1) for score in scores])
        top_k = 5
        top_k_probs, top_k_indices = token_probs.topk(top_k, dim=-1)
        
        for i in range(top_k):
            fig.add_trace(
                go.Scatter(
                    y=top_k_probs[:, 0, i].cpu().numpy(),
                    mode='lines+markers',
                    name=f'Top {i+1} token'
                ),
                row=1, col=1
            )

        # Top 10 Token Probabilities (Final Step)
        final_probs = torch.nn.functional.softmax(scores[-1], dim=-1)
        top_k_probs, top_k_indices = final_probs.topk(10)
        top_k_tokens = [self.tokenizer.decode([idx]) for idx in top_k_indices[0]]

        fig.add_trace(
            go.Bar(x=top_k_tokens, y=top_k_probs[0].cpu().numpy(), marker=dict(color="blue")),
            row=1, col=2
        )

        # Temperature Effect on Token Probabilities
        temperatures = [0.5, 1.0, 2.0]
        for temp in temperatures:
            adjusted_probs = torch.nn.functional.softmax(scores[-1] / temp, dim=-1)
            top_k_probs, _ = adjusted_probs.topk(10)
            fig.add_trace(
                go.Bar(x=top_k_tokens, y=top_k_probs[0].cpu().numpy(), name=f'Temp {temp}'),
                row=2, col=1
            )

        # Text Generation Process
        tokens = self.tokenizer.encode(generated_text)
        token_labels = [self.tokenizer.decode([token]) for token in tokens]
        input_length = len(self.tokenizer.encode(input_text))
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(tokens))),
                y=[1] * len(tokens),
                mode='text',
                text=token_labels,
                textposition="top center",
                textfont=dict(size=10),
                name='Generated Tokens'
            ),
            row=2, col=2
        )
        
        fig.add_shape(
            type="line",
            x0=input_length - 1, y0=0,
            x1=input_length - 1, y1=1,
            line=dict(color="Red", width=2, dash="dash"),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text="Text Generation Visualization",
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Generation Steps", row=1, col=1)
        fig.update_yaxes(title_text="Probability", row=1, col=1)
        
        fig.update_xaxes(title_text="Tokens", row=1, col=2)
        fig.update_yaxes(title_text="Probability", row=1, col=2)
        
        fig.update_xaxes(title_text="Tokens", row=2, col=1)
        fig.update_yaxes(title_text="Probability", row=2, col=1)
        
        fig.update_xaxes(title_text="Token Position", row=2, col=2)
        fig.update_yaxes(title_text="", row=2, col=2)

        fig.show()

    def generate_and_visualize(
        self,
        text: List[str],
        generation_type: GenerationType = GenerationType.GREEDY,
        generation_params: Optional[GenerationParameters] = None
    ) -> None:
        """
        Generates text and visualizes the generation process.

        Args:
            text: List of input text strings.
            generation_type: GenerationType enum value for the desired strategy.
            generation_params: Optional GenerationParameters object for fine-tuning.
        """
        try:
            generated_text, scores = self.generate_text(text, generation_type, generation_params)
            print(f"{generation_type.value.capitalize()} Search:", generated_text)

            if scores:
                self.visualize_generation(scores, text[0], generated_text[0])
            else:
                print("No scores available for visualization.")
        except Exception as e:
            print(f"Error during generation and visualization: {e}")

# Example usage
if __name__ == "__main__":
    model_name = "gpt2"  # You can change this to any other model supported by Hugging Face
    pipeline = TextGenerationPipeline(model_name)

    input_text = ["Once upon a time, in a land far away,"]
    
    # Generate text using different strategies
    for gen_type in GenerationType:
        print(f"\nGenerating text using {gen_type.value}:")
        pipeline.generate_and_visualize(input_text, gen_type)

    # Custom generation parameters
    custom_params = GenerationParameters(
        max_length=100,
        num_beams=3,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.2,
        num_return_sequences=3
    )

    print("\nGenerating text with custom parameters:")
    pipeline.generate_and_visualize(input_text, GenerationType.SAMPLING, custom_params)