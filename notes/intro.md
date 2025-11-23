# Introduction to Reinforcement Fine-Tuning (RFT) for LLMs

Reinforcement Fine-Tuning (RFT) is a training approach that improves an LLM’s behavior using feedback-driven optimization instead of purely supervised labels.  
The goal is to make models *align better with desired behaviors* such as correctness, truthfulness, reasoning, writing style, or safety.

Traditional supervised fine-tuning teaches a model by providing the *correct* response.  
RFT instead rewards the model for producing *better* responses according to a reward function or evaluator.

This creates two advantages:
1. **Flexibility** – You don’t need perfect “gold” answers.  
2. **Optimization toward preferences** – The model learns the *style* and *quality signals* you value.

Modern LLM systems (OpenAI, Anthropic, Google) rely heavily on reinforcement-driven methods to create models that follow instructions reliably.

## Why RFT is Important
- It allows tuning with noisy, subjective, or heuristic feedback.  
- You can optimize for custom constraints (conciseness, safety, tone).  
- It supports rapid iteration without collecting large supervised datasets.  
- It aligns models with human or automated preference systems.

## Where GRPO Fits In
Traditional RLHF (Reinforcement Learning from Human Feedback) requires:
- a reward model  
- a reference model  
- PPO optimization  
- expensive compute  

GRPO (Generative Rejection-based Policy Optimization) simplifies this dramatically by:
- removing the reference model  
- not requiring a value function or critic network  
- using simpler updates that work well for LLMs

This makes reinforcement fine-tuning significantly more accessible for small teams and individual developers.

## What This Repo Covers
This repository follows the DeepLearning.ai course **“Reinforcement Fine-Tuning LLMs with GRPO”** and focuses on:

- Core ideas behind RFT  
- Implementing GRPO components  
- Building reward functions  
- Training pipeline design  
- Running real experiments  
- Evaluating behavior improvements

The repo is structured to help you learn by doing:
- Notes for each lesson  
- Notebooks for interactive exploration  
- Modular source code for policy, rewards, and training  
- Experiment folders to track results like a researcher

You will gradually build a functional GRPO training loop while working through the course.

