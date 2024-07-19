import torch
import torch.nn as nn
from typing import Tuple, Optional

class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, bidirectional: bool = False):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.D = 2 if bidirectional else 1

        # Initialize weight matrices and biases
        self.W_ii = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_if = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_ig = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_io = nn.Parameter(torch.Tensor(input_dim, hidden_dim))

        self.W_hi = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))

        self.b_ii = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_if = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_io = nn.Parameter(torch.Tensor(hidden_dim))

        self.b_hi = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_dim))

        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, x: torch.Tensor, 
                h0: Optional[torch.Tensor] = None, 
                c0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        try:
            batch_size, seq_len, _ = x.size()
            if h0 is None:
                h0 = torch.zeros(self.num_layers * self.D, batch_size, self.hidden_dim).to(x.device)
            if c0 is None:
                c0 = torch.zeros(self.num_layers * self.D, batch_size, self.hidden_dim).to(x.device)

            ht = h0[0]
            ct = c0[0]
            output = []

            for t in range(seq_len):
                xt = x[:, t, :]
                
                it = torch.sigmoid(xt @ self.W_ii + self.b_ii + ht @ self.W_hi + self.b_hi)
                ft = torch.sigmoid(xt @ self.W_if + self.b_if + ht @ self.W_hf + self.b_hf)
                gt = torch.tanh(xt @ self.W_ig + self.b_ig + ht @ self.W_hg + self.b_hg)
                ot = torch.sigmoid(xt @ self.W_io + self.b_io + ht @ self.W_ho + self.b_ho)
                
                ct = ft * ct + it * gt
                ht = ot * torch.tanh(ct)
                
                output.append(ht)

            output = torch.stack(output, dim=1)
            
            if self.bidirectional:
                # Implement bidirectional processing here
                pass

            return output, (ht, ct)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise

def lstm_forward(input_tensor: torch.Tensor, 
                 h0: Optional[torch.Tensor] = None, 
                 c0: Optional[torch.Tensor] = None, 
                 input_dim: int = 10, 
                 hidden_dim: int = 20, 
                 num_layers: int = 1, 
                 bidirectional: bool = False) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    try:
        lstm = LSTM(input_dim, hidden_dim, num_layers, bidirectional=True)
        print("LSTM initialized successfully",lstm)
        output, (hn, cn) = lstm(input_tensor, h0, c0)
        return output, (hn, cn)
    except Exception as e:
        print(f"An error occurred in lstm_forward: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    N, seq, dim_input = 32, 10, 512
    input_tensor = torch.randn(N, seq, dim_input)
    output, (hn, cn) = lstm_forward(input_tensor, input_dim=dim_input,hidden_dim=256,bidirectional=True)
    print(f"Output shape: {output.shape}")
    print(f"hn shape: {hn.shape}")
    print(f"cn shape: {cn.shape}")