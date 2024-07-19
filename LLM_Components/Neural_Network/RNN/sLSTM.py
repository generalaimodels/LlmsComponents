import torch
import torch.nn as nn
from typing import Tuple, Optional
from math import exp, log


class SLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = False,
        bidirectional: bool = False,
    ) -> None:
        super(SLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of sLSTM.

        Args:
            input (torch.Tensor): Input tensor of shape (N, seq, dim_input) if batch_first=True,
                                  else (seq, N, dim_input)
            hx (Optional[Tuple[torch.Tensor, torch.Tensor]]): Initial hidden state and cell state

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output and (hidden state, cell state)
        """
        if self.batch_first:
            input = input.transpose(0, 1)

        seq_len, batch_size, _ = input.size()

        if hx is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=input.device,
            )
            c_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=input.device,
            )
        else:
            h_0, c_0 = hx

        output = []
        h_t = h_0[0]
        c_t = c_0[0]
        n_t = torch.ones_like(c_t)
        m_t = torch.zeros_like(c_t)

        for t in range(seq_len):
            x_t = input[t]
            gates = (
                torch.mm(x_t, self.weight_ih.t())
                + self.bias_ih
                + torch.mm(h_t, self.weight_hh.t())
                + self.bias_hh
            )
            i_t, f_t, z_t, o_t = gates.chunk(4, 1)

            i_t = torch.exp(i_t)
            f_t = torch.exp(f_t)
            z_t = torch.tanh(z_t)
            o_t = torch.sigmoid(o_t)

            m_t = torch.max(torch.log(f_t) + m_t, torch.log(i_t))
            i_t_prime = torch.exp(torch.log(i_t) - m_t)
            f_t_prime = torch.exp(torch.log(f_t) + m_t.roll(1, 0) - m_t)

            c_t = f_t_prime * c_t + i_t_prime * z_t
            n_t = f_t_prime * n_t + i_t_prime
            h_tilde = c_t / n_t
            h_t = o_t * h_tilde

            output.append(h_t)

        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(0, 1)

        return output, (h_t, c_t)

def test_slstm() -> None:
    """Test function for SLSTM."""
    try:
        batch_size = 32
        seq_len = 10
        input_size = 512
        hidden_size = 512
        num_layers = 1

        slstm = SLSTM(input_size, hidden_size, num_layers, batch_first=True)
        input_tensor = torch.randn(batch_size, seq_len, input_size)

        output, (hn, cn) = slstm(input_tensor)
        print(output.shape, hn.shape, cn.shape)

        assert output.shape == (
            batch_size,
            seq_len,
            hidden_size,
        ), f"Expected output shape: {(batch_size, seq_len, hidden_size)}, but got: {output.shape}"
        assert hn.shape == (
            num_layers,
            batch_size,
            hidden_size,
        ), f"Expected hn shape: {(num_layers, batch_size, hidden_size)}, but got: {hn.shape}"
        assert cn.shape == (
            num_layers,
            batch_size,
            hidden_size,
        ), f"Expected cn shape: {(num_layers, batch_size, hidden_size)}, but got: {cn.shape}"

        print("SLSTM test passed successfully!")
    except AssertionError as e:
        print(f"SLSTM test failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

test_slstm()