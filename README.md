# xtream AI Assignment

### Italian Power Load

**Problem type**: time series forecasting

**Dataset description**: [Power Load readme](./datasets/italian-power-load/README.md)

It is your first day in the office and your first project is about time series forecasting.
Your customer is Zap Inc, an imaginary Italian utility: they will provide you with the daily Italian Power Load from 2006 to 2022.
Marta, a colleague of yours, provides you with a wise piece of advice: be careful about 2020, it was a very strange year...

#### Challenge 1

Zap Inc asks you for a complete report about the main feature of the power load series.
The report should be understandable by a layman, but it should also provide enough details to be useful for a data scientist.
**Create a Jupyter notebook to answer their query.**

#### Challenge 2

Then, your first forecasting model.
**You are asked to develop a long-term model to predict the power load 1 year ahead.**
Disregard 2020, 2021, and 2022: use 2019 as test.
Another piece of advice from your colleague Marta.
The managers at Zap Inc are not AI experts, so they want to know how accurate your model is and why they should trust it.
Be sure to answer their concerns in your notebook.

#### Challenge 3

Long-term was great, but what about short term?
**Your next task is to create a short-term model to predict the power load 1 day ahead.**
Disregard 2020, 2021, and 2022: use 2019 as test.
Keep in mind Marta's advice from the previous challenge!

#### Challenge 4

Finally, production trial.
**Pick one of your models and develop and end-to-end pipeline to train and evaluate it on 2020 and 2021.**
Again, your good friend Marta has some suggestion for you. It looks like Luca, the new CTO at Zap, is a bit of a nerd.
And he wants all the production code to be clean, well-structured, and easily maintanable.
You'd better not to disappoint him!

#### Challenge 5

Zap Inc is not impressed by the performance of your model in 2020. You should defend your results.
**Create a notebook to comment and explain the performance of your model in 2020.**

## How to run

![iguana](https://iguanacontrol.com/wp-content/uploads/2020/09/iguana-control-what-to-do.jpg)

### Prerequisites

At the top level of the repo, you will find a [`Justfile`](./justfile). This file contains recipes that are run with [`just`](https://github.com/casey/just) - basically Make, but rewritten in Rust ™️. It is handy, but below you can also find the steps to run the code without it.

1. Ensure you have installed [`PDM`](https://pdm.fming.dev/latest/#installation), a Python dependency manager (everyone gets a package manager). I installed it with `brew`

2. [Install `just`](https://github.com/casey/just#installation). On macOS, you can install it with homebrew:

```bash
brew install just
```

3. Install all the necessary dependencies (also optional and development ones) with the following:

```bash
just install
```

Alternatively, if you do not want to use `just`:

```
pdm install --dev
pdm run pre-commit install --install-hooks
```

Under each of the paragraphs below, you will find the instructions to run each step individually, both with and without `just`.

### Challenge 2

You can click on the badge below to open the notebook in Colab and run all cells.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baggiponte/xtream-ai-assignment/blob/main/notebooks/challenge_2-forecast.ipynb)

If you would like to run the notebook locally, you can use do one of the following:

```bash
# with just
just lab

# without just
pdm run jupyter-lab
```

This command will launch a JupyterLab session. Open the notebook [`./notebooks/challenge_2-forecast.ipynb`](./notebooks/challenge_2-forecast.ipynb) and run all cells.

### Challenge 4

The training script is located under [`./scripts/train.py`](./scripts/train.py) and can be run as follows:

```bash
# with just
just train

# without just
pdm run python scripts/train.py
```

The script accepts the following parameters:

```
usage: train.py [-h] [--training-window TRAINING_WINDOW] [--forecasting-horizon FORECASTING_HORIZON]
                [--validation-strategy VALIDATION_STRATEGY]

Powerload forecasting pipeline

options:
  -h, --help            show this help message and exit
  --training-window TRAINING_WINDOW
                        Cross-validation training window, in days (default: 365*10)
  --forecasting-horizon FORECASTING_HORIZON
                        Cross-validation forecasting horizon, in days (default: 365)
  --validation-strategy VALIDATION_STRATEGY
                        Cross-validation strategy: 'rolling' or 'expanding' (default: 'rolling')
```
