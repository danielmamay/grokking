# Grokking

An implementation of the OpenAI 'Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets' paper in PyTorch.

<img src="figures/Figure_1_left_accuracy.png" height="300"> <img src="figures/Figure_1_left_loss.png" height="300">

## Installation

* Clone the repo and cd into it:
    ```bash
    git clone https://github.com/danielmamay/grokking.git
    cd grokking
    ```
* Use Python 3.9 or later:
    ```bash
    uv sync
    ```

## Usage

The project uses [Weights & Biases](https://wandb.ai/site) to keep track of experiments. Run `uv run wandb login` to use the online dashboard, or `uv run wandb offline` to store the data on your local machine.

* To run a single experiment using the [CLI](grokking/cli.py):
    ```bash
    uv run wandb login
    uv run python grokking/cli.py
    ```

* To run a grid search using W&B Sweeps:
    ```bash
    uv run wandb sweep sweep.yaml
    uv run wandb agent {entity}/grokking/{sweep_id}
    ```

## References

Code:

* [openai/grok](https://github.com/openai/grok)

Paper:

* [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)
