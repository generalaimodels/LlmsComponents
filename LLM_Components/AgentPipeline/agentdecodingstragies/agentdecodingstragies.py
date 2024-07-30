import logging
from typing import Any, Dict, List, Callable, Optional, Tuple,Union
from transformers import AutoModelForCausalLM, AutoTokenizer,set_seed
import torch


logging.basicConfig(level=logging.INFO)
class AgentindividualDecodingStrategies:
    def __init__(self, 
                model:AutoModelForCausalLM,
                tokenizer:AutoTokenizer):
        self.tokenizer = model
        self.model =tokenizer
        
    def generate(self, 
                 input_text: str, 
                 max_length: int = 100, 
                 num_return_sequences: int = 1, 
                 **kwargs) -> List[str]:
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            **kwargs
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def temperature_sampling(self, 
                             input_text: str, 
                             temperature: float = 1.0, 
                             **kwargs) -> List[str]:
        return self.generate(
            input_text,
            do_sample=True,
            temperature=temperature,
            **kwargs
        )

    def top_p_sampling(self, 
                       input_text: str, 
                       top_p: float = 0.9, 
                       **kwargs) -> List[str]:
        return self.generate(
            input_text,
            do_sample=True,
            top_p=top_p,
            **kwargs
        )

    def top_k_sampling(self, 
                       input_text: str, 
                       top_k: int = 50, 
                       **kwargs) -> List[str]:
        return self.generate(
            input_text,
            do_sample=True,
            top_k=top_k,
            **kwargs
        )

    def beam_search(self, 
                    input_text: str, 
                    num_beams: int = 5, 
                    **kwargs) -> List[str]:
        return self.generate(
            input_text,
            num_beams=num_beams,
            **kwargs
        )

    def diverse_beam_search(self, 
                            input_text: str, 
                            num_beams: int = 5, 
                            num_beam_groups: int = 2, 
                            diversity_penalty: float = 1.0, 
                            **kwargs) -> List[str]:
        return self.generate(
            input_text,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            **kwargs
        )

    def constrained_beam_search(self, 
                                input_text: str, 
                                force_words: List[str], 
                                **kwargs) -> List[str]:
        force_words_ids = [self.tokenizer(word, add_special_tokens=False).input_ids for word in force_words]
        
        def prefix_allowed_tokens_fn(batch_id, input_ids):
            return force_words_ids[batch_id % len(force_words_ids)]

        return self.generate(
            input_text,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            **kwargs
        )

    def sequence_to_sequence(self, 
                             input_text: str, 
                             **kwargs) -> List[str]:
        return self.generate(
            input_text,
            encoder_no_repeat_ngram_size=2,
            no_repeat_ngram_size=2,
            **kwargs
        )

    def biased_decoding(self, 
                        input_text: str, 
                        bias_words: Dict[str, float], 
                        **kwargs) -> List[str]:
        sequence_bias = {
            tuple(self.tokenizer.encode(word, add_special_tokens=False)): score
            for word, score in bias_words.items()
        }
        return self.generate(
            input_text,
            sequence_bias=sequence_bias,
            **kwargs
        )

    def length_penalty_decoding(self, 
                                input_text: str, 
                                length_penalty: float = 1.0, 
                                **kwargs) -> List[str]:
        return self.generate(
            input_text,
            length_penalty=length_penalty,
            **kwargs
        )



class AgentTextGenerator:
    def __init__(self,
                  model:AutoModelForCausalLM, 
                  tokenizer=AutoTokenizer,
                  seed_value: int = 10) -> None:
        self.tokenizer = tokenizer
        self.model =model
        set_seed(seed_value)

    def generate_text(self, inputs: list[Dict[str, Any]], generate_kwargs: Optional[Dict[str, Any]] = None) -> str:
        if generate_kwargs is None:
            generate_kwargs = {}

        outputs = self.model.generate(**inputs, **generate_kwargs)
        decoded_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return decoded_text

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
        return self.generate_multiple_texts(inputs, generate_kwargs)

    def non_diverse_summary(self, inputs: Dict[str, Any], max_length: int = 100, num_beams: int = 4,
                            num_return_sequences: int = 2) -> List[str]:
        generate_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "num_return_sequences": num_return_sequences
        }
        return self.generate_multiple_texts(inputs, generate_kwargs)

    def generate_multiple_texts(self, inputs: Dict[str, Any], generate_kwargs: Dict[str, Any]) -> List[str]:
        outputs = self.model.generate(**inputs, **generate_kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)




class TextGenerator:
    def __init__(self,
                  model:AutoModelForCausalLM,
                  tokenizer:AutoTokenizer,
                  seed_value: int = 10) -> None:
        self.tokenizer = tokenizer
        self.model = model
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

class DecodingStrategies:
    def __init__(self,
        model:AutoModelForCausalLM,
        tokenizer:AutoTokenizer,
        seed:int=42):
        self.tokenizer =tokenizer
        self.model = model
        self.seed=seed
        set_seed(seed=self.seed)
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




# def main() -> None:
#     model_name = "bigscience/bloomz-560m"
#     input_sentence = "Once upon a time, in a land far, far away"
#     input_ids = AutoTokenizer.from_pretrained(model_name).encode(input_sentence, return_tensors='pt')

#     inputs = {"input_ids": input_ids}

#     # Initialize TextGenerator with the desired model
#     text_generator = TextGenerator(model_name)

#     # Generate text with different strategies
#     generated_text = text_generator.generate_text(inputs)
#     logging.info(f"Generated Text: {generated_text}")

#     generated_text_with_min_length = text_generator.generate_text(inputs, {"min_length": 3})
#     logging.info(f"Generated Text (with min length): {generated_text_with_min_length}")

#     generate_kwargs = {"max_new_tokens": 10, "do_sample": True, "temperature": 1.0, "num_return_sequences": 2}
#     generated_texts = text_generator.generate_multiple_texts(inputs, generate_kwargs)
#     for idx, text in enumerate(generated_texts):
#         logging.info(f"Generated Text {idx + 1}: {text}")

#     # Demonstrate diverse summary generation
#     diverse_summary_texts = text_generator.diverse_summary(inputs)
#     for idx, text in enumerate(diverse_summary_texts):
#         logging.info(f"Diverse Summary {idx + 1}: {text}")

#     # Demonstrate non-diverse summary generation
#     non_diverse_summary_texts = text_generator.non_diverse_summary(inputs)
#     for idx, text in enumerate(non_diverse_summary_texts):
#         logging.info(f"Non-Diverse Summary {idx + 1}: {text}")


# if __name__ == "__main__":
#     main()