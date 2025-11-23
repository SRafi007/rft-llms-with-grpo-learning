# GRPO Basics (Generative Rejection-based Policy Optimization)

GRPO is a simplified reinforcement fine-tuning algorithm designed specifically for LLMs.  
It focuses on stability, efficiency, and reducing the infrastructure complexity normally required by RLHF or PPO-based systems.

Where classical PPO deals with continuous control environments and needs value functions, GRPO operates directly on **generated text sequences** and uses a rejection-based update strategy.

## Core Idea
The model generates multiple responses for a prompt, evaluates them with a reward function, and then updates the model so that higher-reward responses become more probable.

This eliminates the need for:
- value networks
- advantage estimators
- complex PPO clipping
- reference models with KL penalties

Instead, GRPO uses a simpler objective that works well empirically for LLMs.

## GRPO Pipeline (High-Level)
1. **Prompt Sampling**  
   A batch of prompts is fed to the model.

2. **Response Generation**  
   For each prompt, the model samples several candidate outputs.

3. **Reward Computation**  
   Each output is scored using:
   - a learned reward model  
   - rule-based scoring  
   - heuristics  
   - or custom evaluators

4. **Filtering or Reweighting**  
   Higher-quality samples are kept or weighted more strongly.

5. **Policy Update**  
   The model is updated so that high-reward responses gain probability.

6. **Repeat**  
   Over iterations, the model gradually aligns with your desired behavior.

## How GRPO Differs from PPO/RLHF
| Feature | RLHF (PPO) | GRPO |
|--------|-------------|------|
| Value function | Required | Not needed |
| Reference model | Required | Not needed |
| KL penalty | Explicit | Implicit via sample diversity |
| Complexity | High | Much simpler |
| Stability | Often tricky | Empirically stable |
| Compute cost | Heavy | Moderate/light |

GRPO significantly reduces the overhead of reinforcement tuning for LLMs.

## Why GRPO Works Well for LLMs
- Text generation produces multiple samples easily.  
- Reward distributions allow simple preference-based updates.  
- LLMs benefit from large-batch stochasticity rather than exact advantage estimates.  
- Eliminating the critic simplifies the overall system dramatically.  
- Works well even with noisy rewards and diverse prompt sets.

## Mathematical Intuition (Simplified)
GRPO maximizes:

**Expected reward-weighted log-probability of good responses**

\[
\mathcal{L} = -\mathbb{E}_{\text{samples}}[ w_i \cdot \log \pi_\theta(y_i | x) ]
\]

Where:
- \(x\) is the prompt  
- \(y_i\) are generated responses  
- \(w_i\) are normalized reward-based weights

This is equivalent to:
- pushing the policy towards high-reward generations  
- pulling it away from low-reward ones

Without the need for advantage estimation or a learned critic.

## What You Will Implement in This Repo
- A sampler that generates multiple responses per prompt  
- A reward function (rule-based and model-based)  
- Reward-weight normalization  
- GRPO training step  
- Evaluation suite  
- Experiment tracking

The notebook and source code structure is designed to make each component intuitive and replaceable.

As you progress through the course, you will gradually fill in:
- `sampler.py`  
- `policy.py`  
- `updates.py`  
- `trainer.py`  
- custom reward functions  
- evaluation scripts

This structured approach helps you understand not just how to *use* GRPO, but how to *implement* and *experiment with* it confidently.

