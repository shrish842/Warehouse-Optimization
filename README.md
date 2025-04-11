# Warehouse Layout Optimization using Reinforcement Learning

## Overview

This project explores the use of Reinforcement Learning (RL) to optimize the layout of items in a simulated warehouse. The goal is to minimize the total travel cost associated with retrieving items based on their demand frequency. Higher demand items should ideally be placed closer to the start/retrieval point.

This repository contains a Google Colab / Jupyter Notebook (`Warehouse.ipynb`) implementing:
1.  A custom Gymnasium environment (`WarehouseEnvGeneralized`) simulating the warehouse layout problem.
2.  Training of a Proximal Policy Optimization (PPO) agent using the Stable-Baselines3 library.
3.  Evaluation scripts to assess the trained agent's performance in terms of cost reduction and layout structure compared to the optimal solution.

## Problem Statement

In a warehouse, items are stored at different locations. Each item has a certain demand frequency. The cost of retrieving items is often related to the distance traveled. Assuming locations are arranged linearly from a retrieval point (position 0, 1, 2, ...), the cost can be modeled as the sum over all positions `p`:

`Total Cost = Σ [(p + 1) * demand_frequency(item_at_position_p)]`

The objective is to find an arrangement (layout) of items that minimizes this total cost by placing high-demand items at lower-indexed positions (closer to the start).

## Solution Approach: Reinforcement Learning

We frame this as an RL problem where:
*   **Agent:** A PPO agent from Stable-Baselines3.
*   **Environment:** The custom `WarehouseEnvGeneralized`.
*   **State (Observation):** The current layout of items and their corresponding demand frequencies for the current episode.
*   **Action:** Swapping the positions of any two items in the layout.
*   **Reward:** The negative of the calculated `Total Cost`. The agent aims to maximize this reward, which is equivalent to minimizing the cost.
*   **Generalization:** The environment generates *new random demand frequencies* at the start of each episode (`reset` method), forcing the agent to learn a general strategy for placing high-demand items forward, rather than memorizing a layout for a single demand pattern.

## Environment Details (`WarehouseEnvGeneralized`)

*   **Observation Space:** `gymnasium.spaces.Dict`
    *   `layout`: `spaces.Box(low=0, high=num_items-1, shape=(num_items,), dtype=np.int32)` - Array representing the item ID at each position.
    *   `demands`: `spaces.Box(low=1, high=100, shape=(num_items,), dtype=np.int32)` - Array representing the demand frequency for each item ID (indices match item IDs).
*   **Action Space:** `gymnasium.spaces.Discrete(N)` where `N = num_items * (num_items - 1) // 2`. Each discrete action maps to a unique pair `(i, j)` of layout indices to swap.
*   **Reward:** `-float(cost)` calculated after the swap action.
*   **Episode Termination:** Episodes do not terminate based on reaching a specific goal state in this setup. They typically run for a fixed number of steps during training or evaluation, or are truncated by wrappers.

## Getting Started

### Prerequisites

*   Python (3.8 or later recommended)
*   pip (Python package installer)
*   Git (Optional, for cloning)

### Installation

1.  **Clone the repository (Optional):**
    ```bash
    git clone https://github.com/shrish842/YourRepositoryName.git
    cd YourRepositoryName
    ```
2.  **Set up a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate it:
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install gymnasium "stable-baselines3[extra]" numpy matplotlib scipy
    ```
    *(Note: `stable-baselines3[extra]` includes PyTorch. If running in Colab, some packages might already be installed, but ensure these specific versions are met if needed.)*

## Usage

1.  **Open the Notebook:** Launch `Warehouse.ipynb` using Google Colab, Jupyter Lab, or Jupyter Notebook.
2.  **Run the Cells:** Execute the notebook cells sequentially.
    *   **Environment Definition:** Defines the `WarehouseEnvGeneralized` class.
    *   **Training:**
        *   Sets up the environment and `VecNormalize` wrapper.
        *   Instantiates and trains the PPO agent.
        *   Saves the trained model (`.zip`), normalization statistics (`.pkl`), and TensorBoard logs.
    *   **Evaluation (Cost-Based):**
        *   Loads the saved model and stats.
        *   Runs the agent on multiple new episodes (with different random demands).
        *   Calculates and compares initial, final (agent), and optimal costs.
        *   Generates and saves cost comparison plots (e.g., `evaluation_results_v2.png`).
    *   **Evaluation (Structure-Based):**
        *   Loads the saved model and stats.
        *   Runs the agent on multiple new episodes.
        *   Compares the agent's final layout structure to the optimal (demand-sorted) layout using Kendall's Tau and Top-K item placement accuracy.
        *   Generates and saves layout structure analysis plots (e.g., `evaluation_structure_results_v2.png`).

3.  **Monitor Training (Optional):** Use TensorBoard to view learning progress:
    ```bash
    tensorboard --logdir ./ppo_warehouse_gen_logs_v2/
    ```

## Results

Executing the notebook will:
*   Print training progress and evaluation summaries to the console/output cells.
*   Save the trained agent (`.zip`) and environment statistics (`.pkl`).
*   Create a logs directory (`ppo_warehouse_gen_logs_v2/`) for TensorBoard.
*   Generate `.png` plots visualizing the evaluation results, showing:
    *   Distribution of costs (Initial vs. Agent vs. Optimal).
    *   Cost reduction per episode.
    *   Distribution of layout rank correlation (Kendall's Tau).
    *   Average Top-K item placement accuracy.
    *   Agent's final cost vs. theoretical optimal cost.

## Future Work & Improvements

*   Tune hyperparameters of the PPO agent (learning rate, entropy coefficient, network size, etc.).
*   Experiment with different RL algorithms available in Stable-Baselines3.
*   Implement more sophisticated reward shaping to potentially guide learning better.
*   Extend the environment to include more realistic features (e.g., 2D layout, actual travel distances, item sizes/constraints, multiple retrieval points).
*   Scale the problem to a larger number of items.

## License

<!-- Choose a license and uncomment/update if desired -->
<!-- This project is licensed under the MIT License - see the LICENSE file for details -->
Distributed under the MIT License. See `LICENSE` file for more information (if available).

## Contact

Shrish Agrawal - shrish842 on GitHub
