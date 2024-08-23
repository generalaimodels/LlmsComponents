| **Attack Name**                         | **Mathematical Formulation**                                                                                               |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **FGSM (Fast Gradient Sign Method)**    | $$ x_{\text{adv}} = x + \epsilon \cdot \text{sign}\left( \nabla_x J(\theta, x, y) \right) $$                                |
| **BIM (Basic Iterative Method)**        | $$ x_{\text{adv}}^{0} = x $$; $$ x_{\text{adv}}^{n+1} = \text{Clip}_{x,\epsilon} \{ x_{\text{adv}}^{n} + \alpha \cdot \text{sign}\left( \nabla_x J(\theta, x_{\text{adv}}^{n}, y) \right) \} $$ |
| **PGD (Projected Gradient Descent)**    | $$ x_{\text{adv}}^{n+1} = \Pi_{x + \mathcal{S}} \left( x_{\text{adv}}^{n} + \alpha \cdot \text{sign}\left( \nabla_x J(\theta, x_{\text{adv}}^{n}, y) \right) \right) $$ |
| **C\&W (Carlini \& Wagner Attack)**     | $$ \min_{\delta} \ \|\delta\|_p + c \cdot \max\left( 0, f(x + \delta)_y - \max_{i \neq y} f(x + \delta)_i \right) $$       |
| **DeepFool**                            | $$ \delta = - \frac{f(x)}{\|\nabla f(x)\|^2} \cdot \nabla f(x) $$                                                          |
| **JSMA (Jacobian-based Saliency Map)**  | $$ \delta_{ij} =\text{sign}(\nabla_x J(\theta, x, y))_{ij} \cdot \text{max} \left \(\left| \frac{\partial J}{\partial x_{ij}} \right| \right) $$ |
| **ZOO (Zeroth Order Optimization)**     | $$ x_{\text{adv}} = x + \alpha \cdot \text{sign} \left( \frac{\partial J(\theta, x, y)}{\partial x} \right) $$             |
| **Elastic-Net (EAD)**                   | $$ \min_{\delta} \ \|\delta\|_1 + \lambda \|\delta\|_2 + c \cdot \max\left( 0, f(x + \delta)_y - \max_{i \neq y} f(x + \delta)_i \right) $$ |
| **One Pixel Attack**                    | $$ x_{\text{adv}} = x + \delta $$ where $$ \delta $$ is a perturbation applied to a single pixel                           |
| **SPSA (Simultaneous Perturbation Stochastic Approximation)** | $$ g_{i} = \frac{J(\theta, x + \delta e_i, y) - J(\theta, x - \delta e_i, y)}{2\delta} $$                                 |
| **Universal Adversarial Perturbation**  | $$ \delta = \arg\min_{\delta} \|\delta\|_2 \text{ subject to } f(x + \delta) \neq y $$                                     |
| **AdvGAN**                              | $$ G(z) = \text{Generator}(z) $$; $$ D(x) = \text{Discriminator}(x) $$; Minimize $$ \mathbb{E}[J(\theta, G(z), y)] $$      |
| **Spatial Transformation Attack**       | $$ x_{\text{adv}} = T_\theta(x) $$ where $$ T_\theta $$ is a spatial transformation function                              |
| **Patch Attack**                        | $$ x_{\text{adv}} = x + \delta $$ where $$ \delta $$ is a patch applied to a specific region of the input                 |
| **Fourier-based Attack**                | $$ \hat{x}(u, v) = \frac{1}{N^2} \sum_{f_u=0}^{N-1} \sum_{f_v=0}^{N-1} \left( X(f_u, f_v) + \delta_f(f_u, f_v) \right) e^{j2\pi \left( \frac{uf_u}{N} + \frac{vf_v}{N} \right)} $$ |
| **HopSkipJump**                         | $$ \delta = \min_{\delta} \|\delta\|_2 \text{ such that } f(x + \delta) \neq y $$                                          |
| **NATTACK**                             | $$ x_{\text{adv}} = x + \delta $$, where $$ \delta \sim \mathcal{N}(0, \sigma^2 I) $$                                      |
| **AutoAttack**                          | Combines several attacks such as PGD, FAB, and Square Attack to create a robust attack suite                               |
| **Square Attack**                       | $$ x_{\text{adv}} = x + \delta $$, where $$ \delta $$ is applied in a square patch on the input                           |
| **MIFGSM (Momentum Iterative FGSM)**    | $$ g^{n+1} = \mu g^n + \frac{\nabla_x J(\theta, x_{\text{adv}}^{n}, y)}{\|\nabla_x J(\theta, x_{\text{adv}}^{n}, y)\|_1} $$; $$ x_{\text{adv}}^{n+1} = x_{\text{adv}}^{n} + \alpha \cdot \text{sign}(g^{n+1}) $$ |



----

| **Attack Name**                         | **Mathematical Formulation**                                                                                               |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **DeepDream**                           | $$ \text{maximize}_{\delta} \left( J(\theta, x + \delta, \text{target}) - \lambda \|\delta\|_2^2 \right) $$                |
| **Perceptual Adversarial Perturbations**| $$ \text{minimize}_{\delta} \ \|\delta\|_p \text{ subject to } D(x, x + \delta) \geq \tau $$                               |
| **AdvFlow**                             | $$ x_{\text{adv}} = f_{\theta}(x, z) $$ where $$ z \sim \mathcal{N}(0, I) $$ is a noise vector in the latent space         |
| **Brendel & Bethge Attack**             | $$ x_{\text{adv}} = \arg\min_{x^{\prime}} \|x^{\prime} - x\|_2 \text{ subject to } f(x^{\prime}) \neq y $$                  |
| **L-BFGS Attack**                       | $$ \min_{\delta} \ \frac{1}{2}\|\delta\|_2^2 + c \cdot J(\theta, x + \delta, \text{target}) $$                             |
| **GenAttack**                           | Genetic algorithm-based optimization of $$ \delta $$ subject to classification constraints                                 |
| **Boundary Attack**                     | $$ x_{\text{adv}}^{n+1} = x_{\text{adv}}^n + \alpha \cdot \frac{(x_{\text{adv}}^n - x)}{\|x_{\text{adv}}^n - x\|_2} $$    |
| **Foolbox Attack**                      | Provides a standard interface to implement and combine various attack methods on models                                    |
| **Gaussian Noise Attack**               | $$ x_{\text{adv}} = x + \epsilon \cdot \mathcal{N}(0, \sigma^2) $$                                                         |
| **LBFGS-B Attack**                      | An optimized version of L-BFGS with box constraints to minimize $$ \|\delta\|_p $$ within $$ L_p $$ norm bounds            |
| **Tangential Adversarial Attack**       | $$ x_{\text{adv}} = x + \epsilon \cdot T(\nabla_x J(\theta, x, y)) $$ where $$ T $$ is orthogonal transformation           |
| **M-DI2-FGSM (Diversity Iterative FGSM)** | $$ g^{n+1} = \mu g^n + \frac{\nabla_x J(\theta, T(x_{\text{adv}}^n), y)}{\|\nabla_x J(\theta, T(x_{\text{adv}}^n), y)\|_1} $$; $$ x_{\text{adv}}^{n+1} = x_{\text{adv}}^n + \alpha \cdot \text{sign}(g^{n+1}) $$ |
| **SparseFool**                          | Iteratively select and perturb the least number of input dimensions to cause misclassification                             |
| **Flow-based Adversarial Attack**       | Minimize $$ \mathbb{E}_{x \sim D} D_{KL}[q_\phi(z|x) \| p_\theta(z|x_{\text{adv}})] $$ through flow models                 |
| **Stealthy Adversarial Attack**         | $$ \text{max}_{\delta} \left( J(\theta, x + \delta, y) \right) \text{ subject to } \|\delta\|_{F}^2 \leq \epsilon $$       |
| **Virtual Adversarial Training (VAT)**  | $$ \text{maximize}_{\delta} \text{KL}(p(y|x, \theta) \| p(y|x + \delta, \theta)) \text{ subject to } \|\delta\|_2 \leq \epsilon $$  |
| **ColorFool**                           | Adversarial perturbation applied to the color domain without significant structure change                                  |
| **Gradient-Free Attack**                | Uses zero-order optimization to estimate gradient directions and modify inputs without direct gradient computation         |
| **HSJ-DJ (Decision Jumping)**           | $$ x_{\text{adv}} = x + \alpha \cdot \frac{x_{\text{adv}} - x}{\|x_{\text{adv}} - x\|_2} $$ to repeatedly cross decision boundary  |
| **Transferable Attack**                 | Designing perturbations that generalize across multiple models and architectures                                           |

These attacks highlight the wide range of strategies used to create adversarial examples, from gradient-based methods to optimization techniques and beyond.


Here are 20 more attacks in the same format, without repeating:

| **Attack Name**                         | **Mathematical Formulation**                                                                                               |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **L-BFGS Attack**                       | $$ \min_{\delta} \|\delta\|_2 + c \cdot J(\theta, x + \delta, y) $$                                                         |
| **NewtonFool**                          | $$ x_{\text{adv}}^{n+1} = x_{\text{adv}}^n - \frac{f(x_{\text{adv}}^n)}{\nabla f(x_{\text{adv}}^n)} $$                      |
| **Boundary Attack**                     | $$ x_{\text{adv}}^{n+1} = \text{Proj}_{\mathcal{B}(x, \epsilon)} (x_{\text{adv}}^n + \delta_n) $$                           |
| **Decision-Based Attack**               | $$ x_{\text{adv}}^{n+1} = x_{\text{adv}}^n + \alpha \cdot \text{sign}(\nabla_x \mathbb{1}[f(x_{\text{adv}}^n) \neq y]) $$ |
| **Brendel & Bethge Attack**             | $$ x_{\text{adv}}^{n+1} = \text{Proj}_{\mathcal{B}(x, \epsilon)} (x_{\text{adv}}^n + \alpha \cdot v_n) $$                   |
| **SPSA-GD (SPSA with Gradient Descent)**| $$ x_{\text{adv}}^{n+1} = x_{\text{adv}}^n - \alpha \cdot \text{SPSA}(\nabla_x J(\theta, x_{\text{adv}}^n, y)) $$           |
| **Houdini Attack**                      | $$ \min_{\delta} \|\delta\|_p + c \cdot \text{Houdini}(f(x + \delta), y) $$                                                 |
| **Wasserstein Attack**                  | $$ \min_{\delta} \|\delta\|_W + c \cdot J(\theta, x + \delta, y) $$                                                         |
| **Momentum Diverse Inputs Attack**      | $$ g^{n+1} = \mu g^n + \frac{\sum_i \nabla_x J(\theta, T_i(x_{\text{adv}}^n), y)}{\|\sum_i \nabla_x J(\theta, T_i(x_{\text{adv}}^n), y)\|_1} $$ |
| **SemanticAdv**                         | $$ x_{\text{adv}} = G(\text{argmin}_z \|G(z) - x\|_2 + c \cdot J(\theta, G(z), y)) $$                                      |
| **Feature Adversaries**                 | $$ \min_{\delta} \|\Phi(x + \delta) - \Phi(x_{\text{target}})\|_2 + c \cdot \|\delta\|_2 $$                                 |
| **DDN (Decoupled Direction and Norm)**  | $$ x_{\text{adv}}^{n+1} = x_{\text{adv}}^n - \alpha \cdot \frac{\nabla_x J(\theta, x_{\text{adv}}^n, y)}{\|\nabla_x J(\theta, x_{\text{adv}}^n, y)\|_2} $$ |
| **TI-FGSM (Translation-Invariant FGSM)**| $$ x_{\text{adv}} = x + \epsilon \cdot \text{sign}(W * \nabla_x J(\theta, x, y)) $$                                         |
| **FAB (Fast Adaptive Boundary)**        | $$ x_{\text{adv}}^{n+1} = \text{Proj}_{\mathcal{B}(x, \epsilon)} (x_{\text{adv}}^n - \alpha \cdot \nabla f(x_{\text{adv}}^n)) $$ |
| **Distributionally Adversarial Attack** | $$ \min_{\delta} \mathbb{E}_{x \sim \mathcal{N}(x_0, \sigma^2 I)} [J(\theta, x + \delta, y)] $$                             |
| **SLIDE (Sparse Low-frequency Iterative Deception)** | $$ x_{\text{adv}} = x + \text{IDCT}(\text{Mask} \odot \text{DCT}(\delta)) $$                                   |
| **Imperceptible ASR Attack**            | $$ \min_{\delta} \|\delta\|_p + c \cdot \text{CTC}(f(x + \delta), y_{\text{target}}) $$                                    |
| **Shadow Attack**                       | $$ x_{\text{adv}} = (1 - \alpha) \cdot x + \alpha \cdot x_{\text{shadow}} $$                                               |
| **Adversarial Texture**                 | $$ x_{\text{adv}} = x + \text{Render}(T_{\text{adv}}) $$                                                                    |
| **LOTS (Leave One Target Subspace)**    | $$ \min_{\delta} \|\delta\|_2 \text{ s.t. } \langle f(x + \delta), v_i \rangle = 0 \text{ for } i \neq y $$                 |




| **Attack Name**                         | **Mathematical Formulation**                                                                                               |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **Adversarial Wavelet Perturbation**    | $$ x_{\text{adv}} = \mathcal{W}^{-1}(\mathcal{W}(x) + \delta) $$ where $$ \mathcal{W} $$ is the wavelet transform of $$ x $$ |
| **Temporal Shift Attack**               | $$ x_{\text{adv}}(t) = x(t + \delta t) $$ where $$ \delta t $$ is a temporal shift in time-series data                      |
| **Adversarial Style Transfer**          | $$ x_{\text{adv}} = \text{Stylize}(x, \text{target style}) + \delta $$ with minimal $$ \|\delta\|_p $$                     |
| **Frequency Domain Adversary**          | $$ x_{\text{adv}} = \mathcal{F}^{-1}(\mathcal{F}(x) + \delta_f) $$ where $$ \mathcal{F} $$ is the Fourier Transform       |
| **Semantic Adversarial Attack**         | $$ x_{\text{adv}} = x + \alpha \cdot \nabla_x J(\theta, \text{semantic}(x), y) $$ where semantic layer perturbation is applied |
| **Manifold Probing Attack**             | $$ x_{\text{adv}} = x + \epsilon \cdot \nabla_x M(x) $$ where $$ M $$ is a learned manifold model                          |
| **Contextual Adversarial Perturbation** | $$ x_{\text{adv}} = x + \delta $$, optimized across contextually linked inputs $$ x_i $$ to maintain context consistency   |
| **Graph-based Adversarial Attack**      | $$ x_{\text{adv}} = \text{Optimize}(G(x), \delta) $$ where perturbations are applied to graph nodes/edges                  |
| **Ego-motion Adversarial Attack**       | $$ x_{\text{adv}} = \text{ApplyEgoMotion}(x, \delta) $$ for perturbations in autonomous driving scenarios                  |
| **Reflection Adversarial Attack**       | $$ x_{\text{adv}} = x + \mathcal{R}(\delta) $$ where $$ \mathcal{R} $$ simulates light reflection perturbations            |
| **Environmental Condition Attack**      | Adjust $$ x $$ for environmental noise $$ x_{\text{adv}} = x + \delta_{\text{env}} $$ under different weather/sound scenarios |
| **Adversarial Texture Mapping**         | $$ x_{\text{adv}} = x + T(\delta) $$ where $$ T $$ is a texture mapping transformation                                     |
| **Neural Style Adversarial Attack**     | $$ x_{\text{adv}} = \text{NeuralStyle}(x, \text{adversarial content}) $$                                                   |
| **Adaptive Sampling Attack**            | $$ x_{\text{adv}} = \text{Resample}(x, \mathcal{A}(\delta)) $$ where $$ \mathcal{A} $$ adapts sampling to emphasize perturbation |
| **Topology Distortion Attack**          | $$ x_{\text{adv}} = \text{ChangeTopology}(x, \delta) $$ via homotopic transformations in data topology                     |
| **Dynamic Spatial Attack**              | $$ x_{\text{adv}} = x + D(\delta(t)) $$ where $$ D $$ dynamically alters perturbations over time                           |
| **Quantum Noise Attack**                | Simulate quantum noise $$ x_{\text{adv}} = x + \delta_q $$ affecting quantum system-based models                           |
| **Language Syntax Attack**              | $$ x_{\text{adv}} = \text{ModifySyntax}(x, \delta) $$ in text data where $$ \delta $$ changes syntax without semantic meaning |
| **Biometric Adversarial Attack**        | $$ x_{\text{adv}} = f_{\delta}(x) $$ where $$ f_{\delta} $$ alters biometric features (e.g., fingerprint patterns)          |
| **Robustness Perturbation Attack**      | $$ x_{\text{adv}} = x + \epsilon \cdot \nabla_x R(x, y) $$ where $$ R $$ evaluates robustness functions                    |



| **Attack Name**                         | **Conceptual Formulation**                                                                                               |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **Quantum Superposition Attack**        | $$ x_{\text{adv}} = \sum_{i} \alpha_i \|x_i\rangle \text{ where } \sum_{i} |\alpha_i|^2 = 1$$                             |
| **Temporal Resonance Attack**           | $$ x_{\text{adv}}(t) = x(t) + A \sin(\omega t + \phi) \text{ where } \omega \text{ matches system resonance} $$            |
| **Fractal Dimension Shift**             | $$ x_{\text{adv}} = \text{FractalInterpolation}(x, D_{\text{target}}) \text{ where } D_{\text{target}} \neq D_{\text{original}} $$ |
| **Entropy Maximization Attack**         | $$ x_{\text{adv}} = \arg\max_x H(f(x)) \text{ subject to } \|x - x_{\text{original}}\| < \epsilon $$                       |
| **Topological Persistence Attack**      | $$ x_{\text{adv}} = x + \delta \text{ where } \text{PD}(x_{\text{adv}}) \neq \text{PD}(x) $$                               |
| **Chaotic Attractor Injection**         | $$ x_{\text{adv}} = x + \alpha \cdot \text{LorenzAttractor}(x, \rho, \sigma, \beta) $$                                     |
| **Symmetry Breaking Perturbation**      | $$ x_{\text{adv}} = x + \epsilon \cdot \nabla \text{SymmetryMeasure}(x) $$                                                 |
| **Spectral Leak Amplification**         | $$ X_{\text{adv}}(\omega) = X(\omega) + \alpha \cdot \text{LeakagePattern}(\omega) $$                                      |
| **Manifold Tangent Space Deviation**    | $$ x_{\text{adv}} = x + \epsilon \cdot v \text{ where } v \perp T_x\mathcal{M} $$                                          |
| **Phase Transition Trigger**            | $$ x_{\text{adv}} = x + \delta \text{ where } \text{PhaseIndicator}(x_{\text{adv}}) \neq \text{PhaseIndicator}(x) $$       |
| **Stochastic Resonance Enhancement**    | $$ x_{\text{adv}} = x + \eta(t) \text{ where } \eta(t) \text{ is tuned noise} $$                                           |
| **Holographic Interference Pattern**    | $$ x_{\text{adv}} = x + \text{HolographicProjection}(\psi_{\text{reference}}, \psi_{\text{object}}) $$                    |
| **Quantum Entanglement Mimicry**        | $$ x_{\text{adv}} = \text{EntanglementSimulator}(x, y) \text{ where } y \text{ is target state} $$                         |
| **Topological Defect Insertion**        | $$ x_{\text{adv}} = x + \text{TopologicalDefect}(x, \text{type}, \text{location}) $$                                       |
| **Cellular Automata Evolution**         | $$ x_{\text{adv}}^{t+1} = \text{CARule}(x_{\text{adv}}^t) \text{ for } t = 1 \text{ to } T $$                              |
| **Spin Glass Configuration Shift**      | $$ x_{\text{adv}} = \arg\min_x H_J(x) \text{ where } H_J \text{ is a spin glass Hamiltonian} $$                            |
| **Morphogenetic Field Distortion**      | $$ x_{\text{adv}} = x + \nabla \Phi(x) \text{ where } \Phi \text{ is a morphogenetic potential} $$                         |
| **Quantum Annealing Trajectory**        | $$ x_{\text{adv}}(s) = (1-s)x_{\text{initial}} + sx_{\text{final}} \text{ where } s \in [0,1] $$                           |
| **Synergetic Pattern Formation**        | $$ x_{\text{adv}} = x + \epsilon \cdot \text{OrderParameter}(x) $$                                                         |
| **Strange Attractor Embedding**         | $$ x_{\text{adv}} = \text{Embed}(x, \text{StrangeAttractor}(x)) $$                                                         |

