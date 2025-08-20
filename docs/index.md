# Timepass Trainer

A Python project demonstrating logistic regression training using stochastic gradient descent (SGD) with live progress visualization and rich terminal output.

---

## Overview

Timepass Trainer generates a synthetic classification dataset and trains a logistic regression model incrementally using `SGDClassifier` from scikit-learn. The training process includes:

- Partial fitting over multiple epochs
- Real-time progress bar display using the `rich` library
- Periodic logging of accuracy during training
- Final evaluation metrics displayed in a formatted table

This project highlights incremental learning, interactive terminal UI, and evaluation reporting with ease.

---

## Features

- Synthetic dataset creation (`1000` samples, `20` features, `15` informative)
- Logistic Regression via `SGDClassifier` with random seeding
- Training over `50` epochs with live progress bar and log messages
- Calculation and display of final accuracy, log loss, precision, and recall
- Cleaner output with convergence warnings suppressed
- Built with Python 3.11+, `numpy`, `scikit-learn`, and `rich`

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/timepass-trainer.git
cd timepass-trainer
python3 -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

Run the training script:

```bash
python main.py
```


During execution, observe:

- A progress bar updating with each epoch
- Logged accuracy every 10 epochs
- A summary table of training metrics after completion

---

## Code Overview

- `main.py`: Core training logic including data generation, model training loop, progress reporting, and final metrics display.
- `requirements.txt` & `pyproject.toml`: Define necessary Python dependencies.
- `.gitignore`, `.python-version`: Project environment and ignored files config.

---

## Dependencies

- Python 3.11 or higher
- numpy >= 2.3.2
- scikit-learn >= 1.7.1
- rich >= 14.1.0

Install dependencies using `pip install -r requirements.txt`.

---

## Contributing

Contributions, issue reports, and pull requests are welcome! Please ensure code style conformity and add tests where relevant.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
