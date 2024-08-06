import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Tuple, Optional
import logging
from torch.distributions import kl_divergence
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
        c: float,
        epsilon: float = 1e-8,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
        learning_rate: float = 1e-3,
        kl_target: float = 0.01,
        adaptive_kl_penalty: bool = True,
        entropy_coef: float = 0.01,
        gradient_clip: float = 1.0
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
        self.learning_rate = learning_rate
        self.kl_target = kl_target
        self.adaptive_kl_penalty = adaptive_kl_penalty
        self.entropy_coef = entropy_coef
        self.gradient_clip = gradient_clip

    def policy_gradient_loss(
        self,
        pi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        A: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        try:
            loss_pg = -torch.mean(torch.log(torch.clamp(pi(actions, states), min=self.epsilon)) * A(states, actions))
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
            loss_v = torch.mean((v(states) - R(states)) ** 2)
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
            kl_div = torch.sum(
                torch.where(
                    (pi_old > 0) & (pi_new > 0),
                    pi_old * torch.log((pi_old + self.epsilon) / (pi_new + self.epsilon)),
                    torch.zeros_like(pi_old)
                )
            )
            return kl_div
        except Exception as e:
            logger.error(f"Error in kl_divergence: {e}")
            return torch.tensor(float('inf'))

    def complexity_dependent_loss(self, C: float) -> torch.Tensor:
        try:
            loss_c = self.a * (C ** self.b) + self.c
            return torch.tensor(loss_c)
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
        C: float
    ) -> torch.Tensor:
        try:
            L_pg = self.policy_gradient_loss(pi, A, states, actions)
            L_v = self.value_function_loss(v, R, states)
            D_kl = self.kl_divergence(pi_old, pi_new)
            L_c = self.complexity_dependent_loss(C)

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
        C: float
    ) -> Tuple[Optional[torch.Tensor], float]:
        try:
            params = nn.Parameter(initial_params)
            optimizer = optim.Adam([params], lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

            kl_penalty = 1.0
            prev_loss = float('inf')

            for i in range(self.max_iterations):
                optimizer.zero_grad()
                pi_new = self._update_policy(pi, params, states, actions)
                
                loss, kl_div, entropy_loss = self._compute_total_loss(
                    pi, A, v, R, pi_old, pi_new, states, actions, C, kl_penalty
                )
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(params, self.gradient_clip)
                
                optimizer.step()
                scheduler.step(loss)

                # Adaptive KL penalty adjustment
                if self.adaptive_kl_penalty:
                    kl_penalty = self._adjust_kl_penalty(kl_penalty, kl_div)

                # Check for convergence
                if i > 0 and abs(prev_loss - loss.item()) < self.convergence_threshold:
                    break

                prev_loss = loss.item()

            return params.detach(), loss.item()

        except RuntimeError as e:
            print(f"Runtime error occurred: {e}")
            return None, float('inf')

    def _update_policy(
        self,
        pi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        params: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        return pi(params, states)

    def _compute_total_loss(
        self,
        pi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        A: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        v: Callable[[torch.Tensor], torch.Tensor],
        R: Callable[[torch.Tensor], torch.Tensor],
        pi_old: torch.Tensor,
        pi_new: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        C: float,
        kl_penalty: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        advantage = A(states, actions)
        ratio = torch.exp(pi_new.log_prob(actions) - pi_old.log_prob(actions))
        surrogate_loss = -torch.min(
            ratio * advantage,
            torch.clamp(ratio, 1 - C, 1 + C) * advantage
        ).mean()

        value_loss = torch.nn.functional.mse_loss(v(states), R(states))
        
        kl_div = self.kl_divergence(pi_old, pi_new).mean()
        entropy_loss = -self.entropy_coef * pi_new.entropy().mean()

        total_loss = surrogate_loss + value_loss + kl_penalty * kl_div + entropy_loss

        return total_loss, kl_div, entropy_loss

    def _adjust_kl_penalty(self, kl_penalty: float, kl_div: torch.Tensor) -> float:
        if kl_div > 1.5 * self.kl_target:
            kl_penalty *= 2
        elif kl_div < 0.5 * self.kl_target:
            kl_penalty /= 2
        return kl_penalty



    # def optimize(
    #     self,
    #     initial_params: torch.Tensor,
    #     pi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    #     A: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    #     v: Callable[[torch.Tensor], torch.Tensor],
    #     R: Callable[[torch.Tensor], torch.Tensor],
    #     pi_old: torch.Tensor,
    #     states: torch.Tensor,
    #     actions: torch.Tensor,
    #     C: float
    # ) -> Tuple[Optional[torch.Tensor], float]:
    #     params = nn.Parameter(initial_params)
    #     optimizer = optim.Adam([params])

    #     for i in range(self.max_iterations):
    #         optimizer.zero_grad()
    #         pi_new = self.update_policy(pi, params, states, actions)
    #         loss = self.compute_total_loss(pi, A, v, R, pi_old, pi_new, states, actions, C)
    #         loss.backward()
    #         optimizer.step()

    #         if i > 0 and abs(prev_loss - loss.item()) < self.convergence_threshold:
    #             break

    #         prev_loss = loss.item()

    #     return params.detach(), loss.item()
    # @staticmethod
    # def update_policy(
    #     pi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    #     params: torch.Tensor,
    #     states: torch.Tensor,
    #     actions: torch.Tensor
    # ) -> torch.Tensor:
    #     pi_output = pi(actions, states)
    #     N, seq_length, dimension = pi_output.shape
    #     params_reshaped = params[:N].view(N, 1, 1)
    #     return pi_output * (1 + params_reshaped * 0.01)
    # @staticmethod
    # def update_policy(
    #     pi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    #     params: torch.Tensor,
    #     states: torch.Tensor,
    #     actions: torch.Tensor
    # ) -> torch.Tensor:
    #     return pi(actions, states) * (1 + params.unsqueeze(1).unsqueeze(2) * 0.01)



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from typing import Callable, Optional, Tuple
# from torch.distributions import kl_divergence
# from scipy.stats import entropy
# import numpy as np

# class AdvancedOptimizer:
#     def __init__(
#         self,
#         max_iterations: int = 100,
#         convergence_threshold: float = 1e-5,
#         learning_rate: float = 1e-3,
#         kl_target: float = 0.01,
#         adaptive_kl_penalty: bool = True,
#         entropy_coef: float = 0.01,
#         gradient_clip: float = 1.0
#     ):
#         self.max_iterations = max_iterations
#         self.convergence_threshold = convergence_threshold
#         self.learning_rate = learning_rate
#         self.kl_target = kl_target
#         self.adaptive_kl_penalty = adaptive_kl_penalty
#         self.entropy_coef = entropy_coef
#         self.gradient_clip = gradient_clip

#     def optimize(
#         self,
#         initial_params: torch.Tensor,
#         pi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
#         A: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
#         v: Callable[[torch.Tensor], torch.Tensor],
#         R: Callable[[torch.Tensor], torch.Tensor],
#         pi_old: torch.Tensor,
#         states: torch.Tensor,
#         actions: torch.Tensor,
#         C: float
#     ) -> Tuple[Optional[torch.Tensor], float]:
#         try:
#             params = nn.Parameter(initial_params)
#             optimizer = optim.Adam([params], lr=self.learning_rate)
#             scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

#             kl_penalty = 1.0
#             prev_loss = float('inf')

#             for i in range(self.max_iterations):
#                 optimizer.zero_grad()
#                 pi_new = self._update_policy(pi, params, states, actions)
                
#                 loss, kl_div, entropy_loss = self._compute_total_loss(
#                     pi, A, v, R, pi_old, pi_new, states, actions, C, kl_penalty
#                 )
                
#                 loss.backward()
                
#                 # Gradient clipping for stability
#                 torch.nn.utils.clip_grad_norm_(params, self.gradient_clip)
                
#                 optimizer.step()
#                 scheduler.step(loss)

#                 # Adaptive KL penalty adjustment
#                 if self.adaptive_kl_penalty:
#                     kl_penalty = self._adjust_kl_penalty(kl_penalty, kl_div)

#                 # Check for convergence
#                 if i > 0 and abs(prev_loss - loss.item()) < self.convergence_threshold:
#                     break

#                 prev_loss = loss.item()

#             return params.detach(), loss.item()

#         except RuntimeError as e:
#             print(f"Runtime error occurred: {e}")
#             return None, float('inf')

#     def _update_policy(
#         self,
#         pi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
#         params: torch.Tensor,
#         states: torch.Tensor,
#         actions: torch.Tensor
#     ) -> torch.Tensor:
#         return pi(params, states)

#     def _compute_total_loss(
#         self,
#         pi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
#         A: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
#         v: Callable[[torch.Tensor], torch.Tensor],
#         R: Callable[[torch.Tensor], torch.Tensor],
#         pi_old: torch.Tensor,
#         pi_new: torch.Tensor,
#         states: torch.Tensor,
#         actions: torch.Tensor,
#         C: float,
#         kl_penalty: float
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         advantage = A(states, actions)
#         ratio = torch.exp(pi_new.log_prob(actions) - pi_old.log_prob(actions))
#         surrogate_loss = -torch.min(
#             ratio * advantage,
#             torch.clamp(ratio, 1 - C, 1 + C) * advantage
#         ).mean()

#         value_loss = F.mse_loss(v(states), R(states))
        
#         kl_div = kl_divergence(pi_old, pi_new).mean()
#         entropy_loss = -self.entropy_coef * pi_new.entropy().mean()

#         total_loss = surrogate_loss + value_loss + kl_penalty * kl_div + entropy_loss

#         return total_loss, kl_div, entropy_loss

#     def _adjust_kl_penalty(self, kl_penalty: float, kl_div: torch.Tensor) -> float:
#         if kl_div > 1.5 * self.kl_target:
#             kl_penalty *= 2
#         elif kl_div < 0.5 * self.kl_target:
#             kl_penalty /= 2
#         return kl_penalty








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
        return torch.rand(N, seq_length, dimension)

    def dummy_R(s: torch.Tensor) -> torch.Tensor:
        N, seq_length, dimension = s.shape
        return torch.rand(N, seq_length, dimension)

    N, seq_length, dimension = 10, 5, 3
    dummy_states = torch.rand(N, seq_length, dimension)
    dummy_actions = torch.rand(N, seq_length, dimension)
    dummy_pi_old = torch.rand(N, seq_length, dimension)

    aof = AdvancedObjectiveFunction(0.1, 0.01, 0.001, 1.0, 2.0, 0.5)
    initial_params = torch.zeros(50)
    optimal_params, min_loss = aof.optimize(
        initial_params, dummy_pi, dummy_A, dummy_v, dummy_R,
        dummy_pi_old, dummy_states, dummy_actions, 1
    )
    print(f"optimal_params::{optimal_params}min_loss ::{min_loss }")
