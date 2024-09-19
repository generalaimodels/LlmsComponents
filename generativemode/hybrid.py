
import torch
from torch import nn
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Optional, Dict, Any, Tuple


def hybrid_decoding(
    model: nn.Module,
    input_sequence: torch.Tensor,
    beam_width: int = 5,
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    forced_sequence: Optional[List[int]] = None,
    constraints: Optional[List[int]] = None,
    max_length: int = 50,
    start_token_id: int = 0,
    end_token_id: int = 2,
    device: torch.device = torch.device('cpu')
) -> List[int]:
    """
    Hybrid Advanced Decoding with Multi-Strategy Control.

    Args:
        model (nn.Module): Language generation model.
        input_sequence (torch.Tensor): Input sequence tensor of token IDs.
        beam_width (int): Beam width for beam search.
        top_k (int): Top-k value for sampling.
        top_p (float): Top-p (nucleus) value for sampling.
        temperature (float): Temperature for scaling logits.
        repetition_penalty (float): Penalty for repeated tokens.
        length_penalty (float): Penalty for sequence length.
        forced_sequence (Optional[List[int]]): Sequence of forced tokens.
        constraints (Optional[List[int]]): List of allowed token IDs.
        max_length (int): Maximum length of generated sequence.
        start_token_id (int): Token ID for the start token.
        end_token_id (int): Token ID for the end token.
        device (torch.device): Device to run the model on.

    Returns:
        List[int]: Generated sequence of token IDs.
    """
    # Error handling for inputs
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    if not isinstance(input_sequence, torch.Tensor):
        raise TypeError("input_sequence must be a torch.Tensor")
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0.0, 1.0]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be positive")
    if length_penalty <= 0.0:
        raise ValueError("length_penalty must be positive")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if not isinstance(start_token_id, int) or not isinstance(end_token_id, int):
        raise TypeError("start_token_id and end_token_id must be integers")

    input_sequence = input_sequence.to(device)
    model.to(device)
    model.eval()

    # Initialize
    candidate_sequences: List[List[int]] = [[start_token_id]]  # List of token sequences
    beam_scores: List[float] = [0.0]  # Log probabilities for each beam
    completed_sequences: List[Tuple[float, List[int]]] = []  # Completed sequences with their scores

    for step in range(max_length):
        all_candidates: List[Tuple[float, List[int]]] = []

        for i in range(len(candidate_sequences)):
            seq = candidate_sequences[i]
            seq_score = beam_scores[i]

            # If sequence has ended, no need to expand it
            if seq[-1] == end_token_id:
                completed_sequences.append((seq_score, seq))
                continue

            # Prepare model input
            input_ids = torch.tensor([seq], dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits  # Shape: [batch_size=1, seq_len, vocab_size]

            # Get logits for the last token
            next_token_logits = logits[:, -1, :]
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = apply_repetition_penalty(next_token_logits, seq, repetition_penalty)
            # Apply constraints
            if constraints is not None:
                next_token_logits = apply_constraints(next_token_logits, constraints)
            # Apply forced decoding
            if forced_sequence is not None and step < len(forced_sequence):
                forced_token_id = forced_sequence[step]
                next_token_logits = force_token(next_token_logits, forced_token_id)

            # Apply top-k and top-p filtering
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            filtered_probs = torch.softmax(filtered_logits, dim=-1)  # Shape: [1, vocab_size]

            # Sample tokens
            num_samples = beam_width  # Number of samples per beam
            try:
                sampled_token_ids = torch.multinomial(filtered_probs, num_samples=num_samples)
            except RuntimeError:
                # Handle case when no tokens are available for sampling (all probabilities zero)
                continue

            for k in range(sampled_token_ids.size(1)):
                token_id = int(sampled_token_ids[0, k])
                token_log_prob = float(torch.log(filtered_probs[0, token_id] + 1e-8))
                new_seq = seq + [token_id]
                # Compute new score with length penalty
                length_penalty_coef = ((5 + len(new_seq)) / 6) ** length_penalty
                new_score = (seq_score + token_log_prob) / length_penalty_coef
                all_candidates.append((new_score, new_seq))

        # If no candidates were generated, break the loop
        if not all_candidates:
            break

        # From all candidates, select top beam_width sequences
        all_candidates = sorted(all_candidates, key=lambda x: x[0], reverse=True)
        candidate_sequences = [seq for (_, seq) in all_candidates[:beam_width]]
        beam_scores = [score for (score, _) in all_candidates[:beam_width]]

        # Check for end tokens in candidate sequences
        for i in range(len(candidate_sequences)):
            seq = candidate_sequences[i]
            if seq[-1] == end_token_id:
                completed_sequences.append((beam_scores[i], seq))

        # If we have enough completed sequences, we can stop early
        if len(completed_sequences) >= beam_width:
            break

    # If no completed sequences, use partial sequences
    if not completed_sequences:
        completed_sequences = list(zip(beam_scores, candidate_sequences))

    # Select the best sequence
    completed_sequences = sorted(completed_sequences, key=lambda x: x[0], reverse=True)
    _, best_sequence = completed_sequences[0]
    # Return the best sequence excluding the start token
    return best_sequence[1:]  # Exclude start token


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: List[int],
    repetition_penalty: float
) -> torch.Tensor:
    """
    Apply repetition penalty by penalizing tokens already generated.

    Args:
        logits (torch.Tensor): The logits tensor.
        generated_tokens (List[int]): Tokens generated so far.
        repetition_penalty (float): Penalty factor.

    Returns:
        torch.Tensor: Logits after applying repetition penalty.
    """
    logits = logits.clone()
    for token_id in set(generated_tokens):
        logits[:, token_id] /= repetition_penalty
    return logits


def apply_constraints(
    logits: torch.Tensor,
    constraints: List[int]
) -> torch.Tensor:
    """
    Apply constraints by zeroing out probabilities of tokens not in constraints.

    Args:
        logits (torch.Tensor): The logits tensor.
        constraints (List[int]): Allowed token IDs.

    Returns:
        torch.Tensor: Logits after applying constraints.
    """
    logits = logits.clone()
    vocab_size = logits.size(-1)
    mask = torch.ones(vocab_size, dtype=torch.bool, device=logits.device)
    mask[constraints] = False
    logits[:, mask] = float('-inf')
    return logits


def force_token(
    logits: torch.Tensor,
    forced_token_id: int
) -> torch.Tensor:
    """
    Force the next token to be forced_token_id by setting its logit to a high value.

    Args:
        logits (torch.Tensor): The logits tensor.
        forced_token_id (int): Token ID to force.

    Returns:
        torch.Tensor: Logits after forcing the token.
    """
    logits = torch.full_like(logits, float('-inf'))
    logits[0, forced_token_id] = 0.0  # Assign zero logit to forced token
    return logits


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = float('-inf')
) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits (torch.Tensor): Logits distribution shape (batch_size, vocab_size).
        top_k (int): Keep only top k tokens with highest probability.
        top_p (float): Keep the top tokens with cumulative probability >= top_p.
        filter_value (float): The value to replace filtered logits with.

    Returns:
        torch.Tensor: Filtered logits.
    """
    logits = logits.clone()

    # Top-K filtering
    if top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))  # Safety check
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # Nucleus (Top-P) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to include the token that exceeds top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value

    return logits



class HybridDecoder:
    """
    A hybrid advanced decoder that combines beam search, sampling, and contrastive techniques
    to generate high-quality, contextually relevant sequences.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        beam_width: int = 5,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        forced_sequence: Optional[List[int]] = None,
        constraints: Optional[List[int]] = None,
        max_length: int = 50,
        device: str = "cpu",
    ) -> None:
        """
        Initializes the HybridDecoder.

        Args:
            model (nn.Module): The language generation model.
            tokenizer: The tokenizer associated with the model.
            beam_width (int): Beam width for beam search.
            top_k (int): Top-k value for sampling.
            top_p (float): Top-p (nucleus) probability threshold for sampling.
            temperature (float): Temperature for scaling logits.
            repetition_penalty (float): Penalty for repeated tokens.
            length_penalty (float): Penalty to apply based on the length of the sequence.
            forced_sequence (Optional[List[int]]): Sequence of token IDs to force.
            constraints (Optional[List[int]]): List of token IDs to constrain the output.
            max_length (int): Maximum length of the generated sequence.
            device (str): Device to run the model on ("cpu" or "cuda").
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.forced_sequence = forced_sequence or []
        self.constraints = constraints
        self.max_length = max_length
        self.device = device

    def decode(self, input_ids: torch.Tensor) -> str:
        """
        Generates a sequence based on the input IDs using a hybrid decoding strategy.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.

        Returns:
            str: The decoded sequence as a string.
        """
        try:
            input_ids = input_ids.to(self.device)
            batch_size = input_ids.size(0)
            beam_width = self.beam_width

            # Initialize beams
            sequences = [[list(input_ids[0]), 0.0]]  # List of [sequence, score]
            if self.forced_sequence:
                forced_len = len(self.forced_sequence)
            else:
                forced_len = 0

            for step in range(self.max_length):
                all_candidates = []

                for seq, score in sequences:
                    curr_input = torch.tensor([seq], dtype=torch.long, device=self.device)

                    with torch.no_grad():
                        outputs = self.model(curr_input)
                        logits = outputs.logits
                        next_token_logits = logits[:, -1, :]

                    # Apply temperature
                    next_token_logits = next_token_logits / self.temperature

                    # Apply repetition penalty
                    if self.repetition_penalty != 1.0:
                        token_counts = {}
                        for token_id in seq:
                            token_counts[token_id] = token_counts.get(token_id, 0) + 1
                        for token_id in token_counts:
                            if token_counts[token_id] > 1:
                                next_token_logits[:, token_id] /= self.repetition_penalty

                    # Enforce constraints
                    if self.constraints is not None:
                        mask = torch.ones_like(next_token_logits, dtype=torch.bool)
                        mask[:, self.constraints] = False
                        next_token_logits = next_token_logits.masked_fill(mask, -float('inf'))

                    # Forced decoding
                    if step < forced_len:
                        forced_token = self.forced_sequence[step]
                        next_token_id = forced_token
                    else:
                        # Sampling
                        next_token_logits = self._filter_logits(
                            next_token_logits,
                            top_k=self.top_k,
                            top_p=self.top_p
                        )
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token_id = torch.multinomial(probs, num_samples=1).item()

                    # Length penalty
                    curr_score = score + next_token_logits[0, next_token_id].item()
                    curr_score /= (len(seq) ** self.length_penalty)

                    candidate_seq = seq + [next_token_id]
                    all_candidates.append([candidate_seq, curr_score])

                # Beam pruning
                ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
                sequences = ordered[:beam_width]

                # Check stopping criteria
                if all(seq[-1] == self.tokenizer.eos_token_id for seq, _ in sequences):
                    break

            # Select the best sequence
            best_seq = sequences[0][0]
            decoded_text = self.tokenizer.decode(best_seq, skip_special_tokens=True)
            return decoded_text

        except Exception as e:
            raise RuntimeError(f"Decoding failed: {e}")

    def _filter_logits(self, logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        """
        Filters logits using top-k and nucleus (top-p) filtering.

        Args:
            logits (torch.Tensor): Logits from the model output.
            top_k (int): Keep only top_k tokens with highest probability.
            top_p (float): Keep the top tokens with cumulative probability >= top_p.

        Returns:
            torch.Tensor: Filtered logits.
        """
        try:
            top_k = min(top_k, logits.size(-1))
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1, None]
                logits = torch.where(
                    logits < min_values,
                    torch.full_like(logits, -float('inf')),
                    logits
                )

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                if sorted_indices_to_remove[..., 1:].size(-1) > 0:
                    sorted_indices_to_remove[..., 1:] = (
                        sorted_indices_to_remove[..., :-1].clone()
                    )
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('inf')

            return logits
        except Exception as e:
            raise ValueError(f"Filtering logits failed: {e}")


from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Prepare input
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Initialize the decoder
decoder = HybridDecoder(
    model=model,
    tokenizer=tokenizer,
    beam_width=3,
    top_k=50,
    top_p=0.92,
    temperature=0.7,
    repetition_penalty=1.2,
    length_penalty=1.0,
    forced_sequence=None,
    constraints=None,
    max_length=100,
    device='cpu'
)

# Generate text
output_text = decoder.decode(input_ids)
print(output_text)