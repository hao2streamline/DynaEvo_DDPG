# Model-based Evolutionary Multi-Objective Reinforcement Learning (MOPDERL) for Continuous Control

**Author**: Hao Liu (HL791@exeter.ac.uk)

## Overview

This repository implements a model-based evolutionary multi-objective reinforcement learning (MOPDERL) framework designed for continuous control tasks. The primary goal of this project is to integrate state transition models with evolutionary multi-objective reinforcement learning to solve problems with continuous action spaces, specifically for autonomous driving tasks.

### Key Concepts

- **Reinforcement Learning (RL)**: A machine learning paradigm where agents learn optimal actions through interactions with an environment, receiving rewards and transitioning between states.
- **Model-based Reinforcement Learning (MBRL)**: Uses a learned model of the environment to augment data collected from real-world interactions, improving data efficiency.
- **Evolutionary Reinforcement Learning (ERL)**: Combines evolutionary algorithms with deep reinforcement learning to promote diversity and stability in the learned policies.
- **Dyna Architecture**: A hybrid framework that leverages both real and simulated experiences to update the agent's policy.

---

## Methodology

### Problem Statement
The project explores how to integrate state transition models into evolutionary multi-objective RL algorithms for tasks involving continuous action spaces. The primary application focuses on autonomous driving in a simulated environment.

### MOPDERL Framework

1. **Warm-up Phase**:
   - **Scalarized Weights**: Converts the reward vector into scalar values using fixed weights throughout the training. The number of weights corresponds to the number of objectives.
   - **DDPG Agent Initialization**: Multiple DDPG-based RL agents are initialized, each corresponding to a different scalarization weight. Each agent consists of actors, critics, and their target networks.
   - **Population Formation**: Sub-populations of RL agents are formed, with each agent optimized based on its scalarized reward.
   
2. **Evolutionary Stage**:
   - **NSGA-II Selection**: A multi-objective optimization technique is applied to evolve the population. Sub-populations are merged, and NSGA-II selection is used to generate new individuals.
   - **Distilled Crossover & Mutation**: New individuals are created via crossover and mutation, ensuring diversity and robust exploration of the action space.

3. **Environment Model (MLP) Update**:
   - The MLP environment model is updated using real interaction data. Mean Squared Error (MSE) is used to calculate the next state and reward prediction errors.
   - Both real and simulated experiences are used to update the actor and critic networks in the DDPG framework.

---

## Experimental Setup

### Environment

The project uses the **HighwayEnv** simulation environment, designed for autonomous driving tasks. The environment is integrated with the MLP model to simulate interactions and update experiences.

### Training Process

- **Real and Simulated Experience**: The framework leverages both real-world and simulated experiences to update the actor-critic architecture.
- **MLP Model**: The MLP model predicts the next state and reward, improving the sample efficiency by reducing the reliance on real-world interactions.
- **DDPG Updates**: The actor and critic networks are updated using a combination of real and simulated experiences. Target networks are updated using a soft-update strategy to ensure stability.

### Objective

The ultimate goal is to train MOPDERL agents that can effectively solve the autonomous driving problem by optimizing multiple objectives (e.g., speed, safety) in a continuous action space.

---

## Results

The project has been tested over several episodes in the HighwayEnv simulation. Early results demonstrate:

- **Sample Efficiency**: Leveraging the Dyna framework with simulated experiences improves sample efficiency, reducing the number of real-world interactions required.
- **Diversity & Robustness**: The use of evolutionary algorithms enhances the diversity and robustness of the learned policies, leading to better generalization in unseen scenarios.

---

## Installation

### Requirements

- Python 3.8+
- PyTorch
- TensorBoard
- HighwayEnv

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MOPDERL.git
   cd MOPDERL

2. Install dependencies:
   
    ```bash
     pip install -r requirements.txt


3. Run the training:
   
    ```bash
     python main.py

## Future Work

- **Inverse Reinforcement Learning (IRL)**: To automatically learn reward functions from expert demonstrations, potentially leading to more effective policies.
- **Model Uncertainty**: Incorporating Bayesian Neural Networks to improve the quality of simulated experiences by accounting for uncertainty in the environment model.

---

## Acknowledgements

This project was developed as part of the MSc in Data Science with Artificial Intelligence program at the University of Exeter.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

This README is designed to provide a concise but detailed overview of your project, including its goals, methodology, and how to get started with it on GitHub.
