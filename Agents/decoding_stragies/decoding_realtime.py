from typing import Dict, List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class DecodingStrategies:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def generate(self, 
                 input_text: str, 
                 strategies: Dict[str, Union[bool, float, int, Dict]] = {},
                 max_length: int = 100, 
                 num_return_sequences: int = 1, 
                 **kwargs) -> List[str]:
        inputs = self.tokenizer(input_text, return_tensors="pt")
        generate_kwargs = {
            "max_length": max_length,
            "num_return_sequences": num_return_sequences,
            **kwargs
        }

        # Apply selected strategies
        if "temperature" in strategies:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = strategies["temperature"]
        
        if "top_p" in strategies:
            generate_kwargs["do_sample"] = True
            generate_kwargs["top_p"] = strategies["top_p"]
        
        if "top_k" in strategies:
            generate_kwargs["do_sample"] = True
            generate_kwargs["top_k"] = strategies["top_k"]
        
        if "beam_search" in strategies:
            generate_kwargs["num_beams"] = strategies["beam_search"]
        
        if "diverse_beam_search" in strategies:
            generate_kwargs["num_beams"] = strategies["diverse_beam_search"].get("num_beams", 5)
            generate_kwargs["num_beam_groups"] = strategies["diverse_beam_search"].get("num_beam_groups", 2)
            generate_kwargs["diversity_penalty"] = strategies["diverse_beam_search"].get("diversity_penalty", 1.0)
        
        if "constrained_beam_search" in strategies:
            force_words = strategies["constrained_beam_search"]
            force_words_ids = [self.tokenizer(word, add_special_tokens=False).input_ids for word in force_words]
            
            def prefix_allowed_tokens_fn(batch_id, input_ids):
                return force_words_ids[batch_id % len(force_words_ids)]
            
            generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        
        if "sequence_to_sequence" in strategies:
            generate_kwargs["encoder_no_repeat_ngram_size"] = 2
            generate_kwargs["no_repeat_ngram_size"] = 2
        
        if "biased_decoding" in strategies:
            bias_words = strategies["biased_decoding"]
            sequence_bias = {
                tuple(self.tokenizer.encode(word, add_special_tokens=False)): score
                for word, score in bias_words.items()
            }
            generate_kwargs["sequence_bias"] = sequence_bias
        
        if "length_penalty" in strategies:
            generate_kwargs["length_penalty"] = strategies["length_penalty"]

        outputs = self.model.generate(**inputs, **generate_kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    

import logging
from typing import Any, Dict, List, Callable, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

logging.basicConfig(level=logging.INFO)


class TextGenerator:
    def __init__(self, model_name: str, seed_value: int = 10) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        set_seed(seed_value)

    def generate_text(self, inputs: Dict[str, Any], generate_kwargs: Optional[Dict[str, Any]] = None) -> List[str]:
        if generate_kwargs is None:
            generate_kwargs = {}

        outputs = self.model.generate(**inputs, **generate_kwargs)
        decoded_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_texts

    def get_tokens_as_list(self, word_list: List[str]) -> List[List[int]]:
        tokens_list = []
        for word in word_list:
            tokenized_word = self.tokenizer([word], add_special_tokens=False).input_ids[0]
            tokens_list.append(tokenized_word)
        return tokens_list

    def prefix_allowed_tokens_fn(self, entity: List[int]) -> Callable[[int, List[int]], List[int]]:
        def allowed_tokens_fn(batch_id: int, input_ids: List[int]) -> List[int]:
            if input_ids[-1] == entity[0]:
                return [entity[1].item()]
            elif input_ids[-2] == entity[0] and input_ids[-1] == entity[1]:
                return [entity[2].item()]
            return list(range(self.tokenizer.vocab_size))
        return allowed_tokens_fn

    def diverse_summary(self, inputs: Dict[str, Any], num_beam_groups: int = 2, diversity_penalty: float = 10.0,
                        max_length: int = 100, num_beams: int = 4, num_return_sequences: int = 2) -> List[str]:
        generate_kwargs = {
            "num_beam_groups": num_beam_groups,
            "diversity_penalty": diversity_penalty,
            "max_length": max_length,
            "num_beams": num_beams,
            "num_return_sequences": num_return_sequences
        }
        return self.generate_text(inputs, generate_kwargs)

    def non_diverse_summary(self, inputs: Dict[str, Any], max_length: int = 100, num_beams: int = 4,
                            num_return_sequences: int = 2) -> List[str]:
        generate_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "num_return_sequences": num_return_sequences
        }
        return self.generate_text(inputs, generate_kwargs)

    def all_combinations(self, inputs: Dict[str, Any], decoding_params: List[Dict[str, Any]]) -> List[str]:
        all_generated_texts = []
        for params in decoding_params:
            try:
                generated_texts = self.generate_text(inputs, params)
                all_generated_texts.extend(generated_texts)
            except Exception as e:
                logging.error(f"Error with params {params}: {e}")
        return all_generated_texts

def main() -> None:
    model_name = "bigscience/bloomz-560m"
    input_sentence = "Once upon a time, in a land far, far away"
    input_ids = AutoTokenizer.from_pretrained(model_name).encode(input_sentence, return_tensors='pt')

    inputs = {"input_ids": input_ids}

    # Initialize TextGenerator with the desired model
    text_generator = TextGenerator(model_name)

    # Example decoding parameters for combinations
    decoding_params = [
        {"min_length": 3},
        {"min_new_tokens": 2},
        {"max_new_tokens": 10, "do_sample": True, "temperature": 1.0, "num_return_sequences": 2},
        {"do_sample": True, "top_p": 0.9},
        {"do_sample": True, "top_k": 50},
        {"no_repeat_ngram_size": 2},
        {"encoder_no_repeat_ngram_size": 2},
        {"exponential_decay_length_penalty": (15, 1.6)},
    ]

    # Generate text with all combinations
    all_generated_texts = text_generator.all_combinations(inputs, decoding_params)
    for idx, text in enumerate(all_generated_texts):
        logging.info(f"Generated Text {idx + 1}: {text}")

    # Demonstrate diverse summary generation
    diverse_summary_texts = text_generator.diverse_summary(inputs)
    for idx, text in enumerate(diverse_summary_texts):
        logging.info(f"Diverse Summary {idx + 1}: {text}")

    # Demonstrate non-diverse summary generation
    non_diverse_summary_texts = text_generator.non_diverse_summary(inputs)
    for idx, text in enumerate(non_diverse_summary_texts):
        logging.info(f"Non-Diverse Summary {idx + 1}: {text}")


if __name__ == "__main__":
    main()