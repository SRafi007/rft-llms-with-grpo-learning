# Policy Updates in GRPO

Policy updates are the core engine of GRPO.  
Once we generate samples and compute rewards, the next step is to update the model so that it increases the probability of **high-reward** responses and decreases the probability of **low-reward** ones.

This note explains how policy updates work in GRPO, why they differ from PPO-based RLHF, and how we implement them in this repo.

---

# 1. What Is a Policy in LLM Reinforcement Tuning?

In RFT, the policy **is the language model**.

\[
\pi_\theta(y \mid x)
\]

- \(x\): prompt  
- \(y\): generated response  
- \(\theta\): model parameters

Updating the policy = updating the LLM weights based on a reward-driven objective.

---

# 2. Policy Gradient Basics (LLM Version)

Traditional policy gradient aims to maximize expected reward:

\[
J(\theta) = \mathbb{E}[ R(y) ]
\]

Using the log-derivative trick:

\[
\nabla_\theta J = \mathbb{E}[ R(y) \cdot \nabla_\theta \log \pi_\theta(y|x) ]
\]

For LLMs, this reduces to:

- Find sequences with high rewards  
- Increase their log-probability during training  
- Normalize or scale rewards to keep updates stable

GRPO modifies this principle to make it more stable and simpler.

---

# 3. The GRPO Objective

GRPO uses **reward-weighted log-probabilities**:

\[
\mathcal{L}_{GRPO} = -\sum_i w_i \cdot \log \pi_\theta(y_i | x_i)
\]

Where:

- \(y_i\): sampled responses  
- \(w_i\): normalized reward weights  

This is similar to a **weighted cross-entropy loss**, where better responses have larger weights.

There is **no critic**, **no advantage estimate**, and **no KL penalty**.  
Yet, the method works remarkably well for LLMs because:
- we can cheaply sample multiple outputs  
- reward differences are often large enough  
- normalization prevents instability  

---

# 4. How We Compute Weights \(w_i\)

There are multiple strategies for reward → weight mapping:

### **1. Linear normalization**
\[
w_i = \frac{R_i}{\sum_j R_j}
\]

### **2. Softmax over rewards** (encourages sharper differences)
\[
w_i = \frac{\exp(\alpha R_i)}{\sum_j \exp(\alpha R_j)}
\]

### **3. Rank-based weighting**
Helps with noisy or subjective rewards.

### **4. Rejection filtering**  
Discard low-reward responses entirely and train only on high-reward ones.

In this repo, you will implement several weighting mechanisms so you can compare them experimentally.

---

# 5. Why GRPO Removes KL Penalty

PPO-based RLHF uses a KL penalty to prevent the model from drifting too far from a reference model.

\[
\text{loss} = \text{PPO loss} + \beta \, \text{KL}(\pi \,\Vert\, \pi_\text{ref})
\]

GRPO avoids this because:
- we sample multiple outputs from the same model  
- weight normalization naturally constrains updates  
- high-reward responses are rare unless consistent  
- the process implicitly keeps the model stable

This simplifies the implementation significantly.

---

# 6. Policy Update Steps in Practice

For each training step:

1. **Generate samples**  
   For each prompt, sample \(k\) responses.

2. **Compute rewards**  
   Use rule-based or model-based reward functions.

3. **Convert rewards → weights**  
   Normalize or softmax-transform.

4. **Compute log-probabilities**  
   Get log-prob of each sampled response under the model.

5. **Compute loss**  
   \[
   \mathcal{L} = -\sum_i w_i \cdot \log p(y_i \mid x)
   \]

6. **Backprop & update model**  
   Use AdamW or similar optimizer.

