import logging
from typing import Tuple
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from scipy.stats import entropy
import random
import string
from typing import Tuple
import logging
import random
from typing import List, Any
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticHarmonyModel:
    """
    A model to generate text based on input and compute a semantic harmony score
    between the input and generated output.
    """

    def __init__(self, model_name: str = "gpt2") -> None:
        """
        Initializes the SemanticHarmonyModel with a specified transformer model.
        """
        try:
            logger.info(f"Loading tokenizer for model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"Loading model: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.eval()
            if torch.cuda.is_available():
                self.model.to('cuda')
                logger.info("Model moved to CUDA.")
            else:
                logger.info("CUDA not available. Using CPU.")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise RuntimeError(f"Initialization failed: {e}")

    def generate_text(self, input_text: str, max_length: int = 50) -> str:
        """
        Generates text based on the input text using the transformer model.
        """
        try:
            logger.info("Encoding input text.")
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
            if torch.cuda.is_available():
                input_ids = input_ids.to('cuda')

            logger.info("Generating text.")
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )

            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logger.info("Text generation successful.")
            return generated_text
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise RuntimeError(f"Text generation failed: {e}")

    def compute_semantic_harmony(self, input_text: str, output_text: str) -> float:
        """
        Computes a semantic harmony score between input and output texts.
        """
        try:
            logger.info("Computing semantic harmony score.")
            # Tokenize inputs
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
            output_ids = self.tokenizer.encode(output_text, return_tensors='pt')

            if torch.cuda.is_available():
                input_ids = input_ids.to('cuda')
                output_ids = output_ids.to('cuda')

            # Obtain hidden states
            with torch.no_grad():
                input_hidden = self.model.transformer(input_ids).last_hidden_state
                output_hidden = self.model.transformer(output_ids).last_hidden_state

            # Dynamic contextual weighting: Compute attention alignment
            attention_scores = torch.matmul(input_hidden, output_hidden.transpose(-1, -2))
            attention_weights = torch.softmax(attention_scores, dim=-1)
            dynamic_weight = attention_weights.mean().item()

            # Semantic entropy: Measure uncertainty in the output
            with torch.no_grad():
                output_logits = self.model(output_ids).logits
                output_probs = torch.softmax(output_logits, dim=-1)
            # Detach before converting to NumPy
            output_entropy = entropy(output_probs.detach().cpu().numpy().flatten())

            # Deep probabilistic interdependencies: Compute cosine similarity
            input_vec = input_hidden.mean(dim=1).squeeze().detach().cpu().numpy()
            output_vec = output_hidden.mean(dim=1).squeeze().detach().cpu().numpy()
            cosine_sim = np.dot(input_vec, output_vec) / (
                np.linalg.norm(input_vec) * np.linalg.norm(output_vec) + 1e-10
            )

            # Normalize entropy to [0,1]
            max_entropy = np.log(output_probs.shape[-1])
            normalized_entropy = 1 - (output_entropy / max_entropy)

            # Combine metrics with weights
            score = (
                0.4 * dynamic_weight +
                0.3 * normalized_entropy +
                0.3 * cosine_sim
            )

            # Clamp score to [0,1]
            score = max(0.0, min(1.0, score))
            logger.info(f"Semantic harmony score computed: {score:.4f}")
            return score
        except Exception as e:
            logger.error(f"Error computing semantic harmony: {e}")
            raise RuntimeError(f"Semantic harmony computation failed: {e}")

    def refine_model(self, new_data: Tuple[str, str]) -> None:
        """
        Refines the model using new data to adapt to unseen linguistic patterns.
        """
        try:
            logger.info("Refining model with new data.")
            # Placeholder for self-adjusting statistical techniques
            # This could involve fine-tuning the model on new data
            # or updating internal metrics based on new observations.
            # Implementation depends on specific requirements and data.

            # Example: Fine-tuning on new data (simple demonstration)
            input_text, output_text = new_data
            inputs = self.tokenizer(input_text, return_tensors="pt")
            labels = self.tokenizer(output_text, return_tensors="pt")["input_ids"]

            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                labels = labels.to('cuda')

            self.model.train()
            with torch.no_grad():  # Prevent gradient tracking during refinement
                outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            # Optimizer step would be here, omitted for brevity
            self.model.eval()
            logger.info("Model refinement successful.")
        except Exception as e:
            logger.error(f"Error refining model: {e}")
            raise RuntimeError(f"Model refinement failed: {e}")


# # Example usage:
# if __name__ == "__main__":
#     try:
#         model = SemanticHarmonyModel(model_name="gpt2")
#         input_str = "Once upon a time in a land far, far away"
#         generated_str = model.generate_text(input_str, max_length=30)
#         print(generated_str)
#         score = model.compute_semantic_harmony(input_str, generated_str)
#         print(score)
#         logger.info(f"Input Text: {input_str}")
#         logger.info(f"Generated Text: {generated_str}")
#         logger.info(f"Semantic Harmony Score: {score:.4f}")
#     except Exception as error:
#         logger.error(f"An error occurred during processing: {error}")



class BlackBoxTextGenerator:
    """
    A completely black-box text generation model that processes a string input
    to produce a string output along with a numerical score between 0 and 1.
    
    The internal statistical-probabilistic system is opaque and does not expose
    its inner workings.
    """

    def __init__(self) -> None:
        """
        Initializes the BlackBoxTextGenerator instance.
        """
        try:
            # Initialize any necessary components or configurations here
            # For demonstration purposes, no actual model is loaded
            logging.info("BlackBoxTextGenerator initialized successfully.")
        except Exception as error:
            logging.error(f"Initialization failed: {error}")
            raise

    def generate(self, input_text: str) -> Tuple[str, float]:
        """
        Generates an output string based on the input string and computes a coherence score.

        Args:
            input_text (str): The input string to process.

        Returns:
            Tuple[str, float]: A tuple containing the generated output string and a numerical score between 0 and 1.

        Raises:
            ValueError: If the input_text is empty or not a string.
            Exception: For any unexpected errors during generation.
        """
        try:
            if not isinstance(input_text, str):
                logging.error("Input must be a string.")
                raise ValueError("Input must be a string.")
            
            if not input_text.strip():
                logging.error("Input string is empty.")
                raise ValueError("Input string cannot be empty.")
            
            logging.info(f"Generating output for input: {input_text}")

            # Simulate text generation (Placeholder for actual model logic)
            output_text = self._simulate_text_generation(input_text)

            # Simulate score computation (Placeholder for actual scoring mechanism)
            score = self._simulate_score_computation(input_text, output_text)

            logging.info(f"Generated output: {output_text} with score: {score}")

            return output_text, score

        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during generation: {e}")
            raise

    def _simulate_text_generation(self, input_text: str) -> str:
        """
        Simulates text generation by reversing the input string and adding random characters.

        Args:
            input_text (str): The input string to process.

        Returns:
            str: The generated output string.
        """
        reversed_text = input_text[::-1]
        random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        generated_text = f"{reversed_text}{random_suffix}"
        logging.debug(f"Simulated text generation: {generated_text}")
        return generated_text

    def _simulate_score_computation(self, input_text: str, output_text: str) -> float:
        """
        Simulates the computation of a coherence score between input and output texts.

        Args:
            input_text (str): The original input string.
            output_text (str): The generated output string.

        Returns:
            float: A numerical score between 0 and 1 indicating coherence.
        """
        # Simple heuristic: longer similarity between input and output increases the score
        similarity = self._compute_similarity(input_text, output_text)
        normalized_score = min(max(similarity, 0.0), 1.0)
        logging.debug(f"Simulated score computation: {normalized_score}")
        return normalized_score

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Computes a simple similarity score between two texts based on common characters.

        Args:
            text1 (str): The first text string.
            text2 (str): The second text string.

        Returns:
            float: A similarity score between 0 and 1.
        """
        set1 = set(text1)
        set2 = set(text2)
        common_chars = set1.intersection(set2)
        total_chars = set1.union(set2)
        if not total_chars:
            return 0.0
        similarity_ratio = len(common_chars) / len(total_chars)
        logging.debug(f"Computed similarity ratio: {similarity_ratio}")
        return similarity_ratio

# # Example usage:
# if __name__ == "__main__":
#     try:
#         generator = BlackBoxTextGenerator()
#         input_str = "An iterable-style dataset is an instance of a subclass of IterableDataset that implements the __iter__() protocol, and represents an iterable over data samples. This type of datasets is particularly suitable for cases where random reads are expensive or even improbable, and where the batch size depends on the fetched data."
#         output_str, score = generator.generate(input_str)
#         print(f"Input: {input_str}")
#         print(f"Output: {output_str}")
#         print(f"Score: {score}")
#     except Exception as error:
#         print(f"An error occurred: {error}")

import random
from typing import List


class StringDecomposer:
    """
    A class to decompose a string into randomized chunks, words, and characters
    with reproducible randomness controlled by a seed.
    """

    def __init__(self, input_string: str, seed: int) -> None:
        """
        Initialize the StringDecomposer with the input string and seed.

        :param input_string: The string to be decomposed.
        :param seed: The seed to ensure reproducible randomness.
        :raises ValueError: If input_string is not a string or seed is not an integer.
        """
        if not isinstance(input_string, str):
            raise ValueError("input_string must be a string.")
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer.")

        self.input_string: str = input_string
        self.seed: int = seed
        self.random = random.Random(seed)

    def decompose(self) -> List:
        """
        Perform the decomposition of the input string into chunks, words, and characters.

        :return: A list representing the decomposed structure.
        """
        try:
            chunks = self._split_into_chunks(self.input_string)
            randomized_chunks = self.random.sample(chunks, len(chunks))
            result = [self._split_into_words(chunk) for chunk in randomized_chunks]
            # Flatten the list of lists
            words = [word for sublist in result for word in sublist]
            randomized_words = self.random.sample(words, len(words))
            chars = [self._split_into_characters(word) for word in randomized_words]
            # Flatten the list of lists
            characters = [char for sublist in chars for char in sublist]
            randomized_characters = self.random.sample(characters, len(characters))
            return randomized_characters
        except Exception as e:
            raise RuntimeError(f"An error occurred during decomposition: {e}") from e

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split the text into chunks of random lengths.

        :param text: The text to split.
        :return: A list of chunked strings.
        """
        if not text:
            return []

        min_chunk_size = 2
        max_chunk_size = max(2, len(text) // 2)

        chunks = []
        index = 0
        while index < len(text):
            remaining = len(text) - index
            chunk_size = self.random.randint(min_chunk_size, min(max_chunk_size, remaining))
            chunks.append(text[index:index + chunk_size])
            index += chunk_size
        return chunks

    def _split_into_words(self, chunk: str) -> List[str]:
        """
        Split a chunk into words based on whitespace.

        :param chunk: The chunk to split.
        :return: A list of words.
        """
        return chunk.split()

    def _split_into_characters(self, word: str) -> List[str]:
        """
        Split a word into its constituent characters.

        :param word: The word to split.
        :return: A list of characters.
        """
        return list(word)


# def main():
#     """
#     Example usage of StringDecomposer.
#     """
#     input_str = "This is an example string to demonstrate the decomposition process."
#     seed_number = 42

#     try:
#         decomposer = StringDecomposer(input_str, seed_number)
#         decomposition_result = decomposer.decompose()
#         print("Decomposition Result:")
#         print(decomposition_result)
#     except Exception as error:
#         print(f"Error: {error}")


# if __name__ == "__main__":
#     main()




def characters_to_sentences(characters: List[str], seed: int) -> str:
    """
    Combines a list of characters into sentences using a fixed random seed.

    :param characters: List of single-character strings.
    :param seed: Integer seed for randomization to ensure reproducibility.
    :return: A string consisting of sentences formed from the input characters.
    :raises ValueError: If input is invalid.
    """
    _validate_input(characters, seed)

    rng = random.Random(seed)

    # Shuffle characters to ensure random distribution
    shuffled_chars = characters.copy()
    rng.shuffle(shuffled_chars)

    # Group characters into words
    words = _group_elements(shuffled_chars, rng, min_size=2, max_size=8)
    
    # **Convert each word from a list of characters to a string**
    words = [''.join(word_chars) for word_chars in words]

    # Shuffle words to randomize sentence construction
    rng.shuffle(words)

    # Group words into sentences
    sentences = _group_elements(words, rng, min_size=5, max_size=15)

    # Capitalize first word of each sentence and add a period at the end
    formatted_sentences = []
    for sentence_words in sentences:
        if not sentence_words:
            continue
        sentence = ' '.join(sentence_words)
        sentence = sentence.capitalize() + '.'
        formatted_sentences.append(sentence)

    return ' '.join(formatted_sentences)


def _validate_input(characters: List[Any], seed: Any) -> None:
    """
    Validates the input parameters.

    :param characters: The list to validate.
    :param seed: The seed to validate.
    :raises ValueError: If validation fails.
    """
    if not isinstance(characters, list):
        raise ValueError("Characters input must be a list.")

    if not all(isinstance(c, str) and len(c) == 1 for c in characters):
        raise ValueError("All items in characters list must be single-character strings.")

    if not isinstance(seed, int):
        raise ValueError("Seed must be an integer.")

    if not characters:
        raise ValueError("Characters list cannot be empty.")


def _group_elements(elements: List[Any], rng: random.Random, min_size: int, max_size: int) -> List[List[Any]]:
    """
    Groups elements into sublists based on random sizes.

    :param elements: The list of elements to group.
    :param rng: A random number generator instance.
    :param min_size: Minimum size of each group.
    :param max_size: Maximum size of each group.
    :return: A list of grouped elements.
    """
    groups = []
    index = 0
    total = len(elements)
    while index < total:
        remaining = total - index
        group_size = rng.randint(min_size, max_size)
        group_size = min(group_size, remaining)
        group = elements[index:index + group_size]
        groups.append(group)
        index += group_size
    return groups


# # Example Usage
# if __name__ == "__main__":
#     input_str = "This is an example string to demonstrate the decomposition process."
#     seed_number = 41
#     print("Input String:")
#     print(input_str)

#     try:
#         decomposer = StringDecomposer(input_str, seed_number)
#         decomposition_result = decomposer.decompose()
#         print("Decomposition Result:")
#         print(decomposition_result)
#     except Exception as error:
#         print(f"Error: {error}")
#     sample_characters =   decomposition_result
#     fixed_seed = 40
#     try:
#         result = characters_to_sentences(sample_characters, fixed_seed)
#         print(result)
#     except ValueError as e:
#         print(f"Error: {e}")