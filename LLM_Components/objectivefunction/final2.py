from typing import Callable, Dict, Any, List
import numpy as np

class AdvancedOptimizer:
    def __init__(self, alpha: float, beta: float, gamma: float, a: float, b: float, c: float) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.a = a
        self.b = b
        self.c = c

    def policy_gradient_loss(self, pi: Callable[[Any, Any], float], A: Callable[[Any, Any], float],
                             states: np.ndarray, actions: np.ndarray) -> float:
        try:
            losses: List[float] = [
                np.log(pi(a, s)) * A(s, a) for s, a in zip(states, actions)
            ]
            loss_pg = -np.mean(losses)
            return loss_pg
        except Exception as e:
            print(f"An error occurred in policy_gradient_loss: {e}")
            return float('inf')

    def value_function_loss(self, v: Callable[[Any], float], R: Callable[[Any], float], states: np.ndarray) -> float:
        try:
            losses: List[float] = [
                (v(s) - R(s)) ** 2 for s in states
            ]
            loss_v = np.mean(losses)
            return loss_v
        except Exception as e:
            print(f"An error occurred in value_function_loss: {e}")
            return float('inf')
    
    def kl_divergence(self, pi_old: Dict[Any, float], pi_new: Dict[Any, float]) -> float:
        try:
            kl_div = sum(
                pi_old[x] * np.log(pi_old[x] / pi_new[x])
                for x in pi_old if pi_old[x] > 0 and x in pi_new and pi_new[x] > 0
            )
            return kl_div
        except Exception as e:
            print(f"An error occurred in kl_divergence: {e}")
            return float('inf')

    def complexity_dependent_loss(self, C: float) -> float:
        try:
            loss_c = self.a * (C ** self.b) + self.c
            return loss_c
        except Exception as e:
            print(f"An error occurred in complexity_dependent_loss: {e}")
            return float('inf')

    def compute_total_loss(self, pi: Callable[[Any, Any], float], A: Callable[[Any, Any], float],
                           v: Callable[[Any], float], R: Callable[[Any], float], 
                           pi_old: Dict[Any, float], pi_new: Dict[Any, float],
                           states: np.ndarray, actions: np.ndarray, C: float) -> float:
        try:
            L_pg = self.policy_gradient_loss(pi, A, states, actions)
            L_v = self.value_function_loss(v, R, states)
            D_kl = self.kl_divergence(pi_old, pi_new)
            L_c = self.complexity_dependent_loss(C)
            L_total = L_pg + self.alpha * L_v + self.beta * D_kl + self.gamma * L_c
            return L_total
        except Exception as e:
            print(f"An error occurred in compute_total_loss: {e}")
            return float('inf')
        
