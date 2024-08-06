import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Tuple, Optional
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedObjectiveFunction(nn.Module):
    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        a: float,
        b: float,
        c: torch.Tensor,
        epsilon: float = 1e-8,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.a = a
        self.b = b
        self.c = c
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def policy_gradient_loss(
        self,
        pi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        A: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        try:
            log_pi = torch.log(torch.clamp(pi(actions, states), min=self.epsilon))
            advantages = A(states, actions)
            loss_pg = -torch.mean(log_pi * advantages)
            return loss_pg
        except Exception as e:
            logger.error(f"Error in policy_gradient_loss: {e}")
            return torch.tensor(float('inf'))

    def value_function_loss(
        self,
        v: Callable[[torch.Tensor], torch.Tensor],
        R: Callable[[torch.Tensor], torch.Tensor],
        states: torch.Tensor
    ) -> torch.Tensor:
        try:
            values = v(states)
            returns = R(states)
            loss_v = nn.MSELoss()(values, returns)
            return loss_v
        except Exception as e:
            logger.error(f"Error in value_function_loss: {e}")
            return torch.tensor(float('inf'))

    def kl_divergence(
        self,
        pi_old: torch.Tensor,
        pi_new: torch.Tensor
    ) -> torch.Tensor:
        try:
            kl_div = torch.mean(
                torch.sum(
                    pi_old * (torch.log(pi_old + self.epsilon) - torch.log(pi_new + self.epsilon)),
                    dim=-1
                )
            )
            return kl_div
        except Exception as e:
            logger.error(f"Error in kl_divergence: {e}")
            return torch.tensor(float('inf'))
    


    def complexity_dependent_loss(self, C: torch.Tensor) -> torch.Tensor:
        try:
            # Assuming C has shape (N, seq_len, dim)
            # Compute complexity measure (e.g., L2 norm) along dim axis
            complexity = torch.norm(C, dim=-1)  # Shape: (N, seq_len)
            
            # Compute mean complexity across sequence length
            mean_complexity = torch.mean(complexity, dim=-1)  # Shape: (N,)
            
            # Apply complexity-dependent loss function
            loss_c = self.a * (mean_complexity ** self.b) + self.c
            
            # Ensure the output is a scalar
            return torch.mean(loss_c)
    
        except Exception as e:
            logger.error(f"Error in complexity_dependent_loss: {e}")
            return torch.tensor(float('inf'))
   

    def compute_total_loss(
        self,
        pi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        A: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        v: Callable[[torch.Tensor], torch.Tensor],
        R: Callable[[torch.Tensor], torch.Tensor],
        pi_old: torch.Tensor,
        pi_new: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        
    ) -> torch.Tensor:
        try:
            L_pg = self.policy_gradient_loss(pi, A, states, actions)
            L_v = self.value_function_loss(v, R, states)
            D_kl = self.kl_divergence(pi_old, pi_new)
            L_c = self.complexity_dependent_loss(pi_new)

            L_total = L_pg + self.alpha * L_v + self.beta * D_kl + self.gamma * L_c
            return L_total
        except Exception as e:
            logger.error(f"Error in compute_total_loss: {e}")
            return torch.tensor(float('inf'))

    def optimize(
        self,
        initial_params: torch.Tensor,
        pi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        A: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        v: Callable[[torch.Tensor], torch.Tensor],
        R: Callable[[torch.Tensor], torch.Tensor],
        pi_old: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
       
    ) -> Tuple[Optional[torch.Tensor], float]:
        params = nn.Parameter(initial_params)
        optimizer = optim.Adam([params])

        prev_loss = float('inf')
        for i in range(self.max_iterations):
            optimizer.zero_grad()
            pi_new = self.update_policy(pi, params, states, actions)
            loss = self.compute_total_loss(pi, A, v, R, pi_old, pi_new, states, actions,)
            loss.backward()
            optimizer.step()

            if i > 0 and abs(prev_loss - loss.item()) < self.convergence_threshold:
                break

            prev_loss = loss.item()

        return params.detach(), loss.item()

    @staticmethod
    def update_policy(
        pi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        params: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        pi_output = pi(actions, states)
        N, seq_length, dimension = pi_output.shape
        params_reshaped = params[:N].view(N, 1, 1)
        return pi_output * (1 + params_reshaped * 0.01)

if __name__ == "__main__":
    # Example usage
    def dummy_pi(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        N, seq_length, dimension = a.shape
        return torch.rand(N, seq_length, dimension)

    def dummy_A(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        N, seq_length, dimension = s.shape
        return torch.rand(N, seq_length, dimension)

    def dummy_v(s: torch.Tensor) -> torch.Tensor:
        N, seq_length, dimension = s.shape
        return torch.rand(N, seq_length, 1)

    def dummy_R(s: torch.Tensor) -> torch.Tensor:
        N, seq_length, dimension = s.shape
        return torch.rand(N, seq_length, 1)

    N, seq_length, dimension = 10, 5, 3
    dummy_states = torch.rand(N, seq_length, dimension)
    dummy_actions = torch.rand(N, seq_length, dimension)
    dummy_pi_old = torch.rand(N, seq_length, dimension)

    aof = AdvancedObjectiveFunction(0.1, 0.01, 0.001, 1.0, 2.0, 0.5)
    initial_params = torch.zeros(50)
    optimal_params, min_loss = aof.optimize(
        initial_params, dummy_pi, dummy_A, dummy_v, dummy_R,
        dummy_pi_old, dummy_states, dummy_actions,
    )
    print(f"Optimal parameters: {optimal_params}")
    print(f"Minimum loss: {min_loss}")

    # Evaluate the optimized policy
    optimized_pi = aof.update_policy(dummy_pi, optimal_params, dummy_states, dummy_actions)
    
    # Calculate the final losses
    final_pg_loss = aof.policy_gradient_loss(dummy_pi, dummy_A, dummy_states, dummy_actions)
    final_v_loss = aof.value_function_loss(dummy_v, dummy_R, dummy_states)
    final_kl_div = aof.kl_divergence(dummy_pi_old, optimized_pi)
    final_complexity_loss = aof.complexity_dependent_loss(1.0)
    
    print(f"Final Policy Gradient Loss: {final_pg_loss.item()}")
    print(f"Final Value Function Loss: {final_v_loss.item()}")
    print(f"Final KL Divergence: {final_kl_div.item()}")
    print(f"Final Complexity-Dependent Loss: {final_complexity_loss.item()}")

    # Visualize the optimization process
    import matplotlib.pyplot as plt

    def visualize_optimization(aof, initial_params, dummy_pi, dummy_A, dummy_v, dummy_R,
                               dummy_pi_old, dummy_states, dummy_actions, ):
        losses = []
        params = initial_params.clone()
        for _ in range(100):  # Run for 100 iterations
            pi_new = aof.update_policy(dummy_pi, params, dummy_states, dummy_actions)
            loss = aof.compute_total_loss(dummy_pi, dummy_A, dummy_v, dummy_R, dummy_pi_old, pi_new, dummy_states, dummy_actions,)
            losses.append(loss.item())
            params = aof.optimize(params, dummy_pi, dummy_A, dummy_v, dummy_R, dummy_pi_old, dummy_states, dummy_actions,)[0]

        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Optimization Process')
        plt.xlabel('Iteration')
        plt.ylabel('Total Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.show()

    visualize_optimization(aof, initial_params, dummy_pi, dummy_A, dummy_v, dummy_R,
                           dummy_pi_old, dummy_states, dummy_actions, )

    # Sensitivity analysis
    def sensitivity_analysis(parameter_name, parameter_range):
        losses = []
        for value in parameter_range:
            setattr(aof, parameter_name, value)
            _, loss = aof.optimize(initial_params, dummy_pi, dummy_A, dummy_v, dummy_R,
                                   dummy_pi_old, dummy_states, dummy_actions, )
            losses.append(loss)

        plt.figure(figsize=(10, 6))
        plt.plot(parameter_range, losses)
        plt.title(f'Sensitivity Analysis: {parameter_name}')
        plt.xlabel(parameter_name)
        plt.ylabel('Minimum Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.show()

    # Perform sensitivity analysis for alpha, beta, and gamma
    sensitivity_analysis('alpha', np.linspace(0.01, 1.0, 20))
    sensitivity_analysis('beta', np.linspace(0.001, 0.1, 20))
    sensitivity_analysis('gamma', np.linspace(0.0001, 0.01, 20))



