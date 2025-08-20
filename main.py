# train_scheduler.py
import time
import logging
import warnings
import numpy as np
import random
from datetime import datetime, timedelta
from rich.logging import RichHandler
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score
from sklearn.exceptions import ConvergenceWarning
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn
from rich.table import Table

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    handlers=[RichHandler(rich_traceback=True)]
)
logger = logging.getLogger(__name__)

# ---------------- Console Setup ----------------
console = Console()

# ---------------- Suppress Warnings ----------------
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------- Training Function ----------------
def train_model():
    console.rule("[bold green]🚀 Starting Training[/bold green]")

    # Generate dummy dataset (random every trigger)
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=random.randint(0, 10000)  # random dataset seed
    )

    # SGDClassifier with random seed
    model = SGDClassifier(
        loss="log_loss",
        max_iter=1,
        learning_rate="constant",
        eta0=0.01,
        random_state=random.randint(0, 10000)  # random model seed
    )

    # Training loop with progress bar
    epochs = 50
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Training Logistic Regression", total=epochs)

        for epoch in range(epochs):
            model.partial_fit(X, y, classes=np.unique(y))  # partial fit
            progress.update(task, advance=1)

            if epoch % 10 == 0:
                acc = accuracy_score(y, model.predict(X))
                logger.info(f"Epoch {epoch} - Accuracy={acc:.4f}")

    # Final evaluation
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    loss = log_loss(y, model.predict_proba(X))
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    # Add small random noise to simulate metric variability
    acc += random.uniform(-0.02, 0.02)
    loss += random.uniform(-0.05, 0.05)
    precision += random.uniform(-0.02, 0.02)
    recall += random.uniform(-0.02, 0.02)

    # Clamp values between 0 and 1 where needed
    acc = max(0, min(1, acc))
    precision = max(0, min(1, precision))
    recall = max(0, min(1, recall))
    loss = max(0, loss)

    # Results table
    table = Table(title="📊 Training Results")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Accuracy", f"{acc:.4f}")
    table.add_row("Loss", f"{loss:.4f}")
    table.add_row("Precision", f"{precision:.4f}")
    table.add_row("Recall", f"{recall:.4f}")
    table.add_row("Samples", str(X.shape[0]))
    table.add_row("Features", str(X.shape[1]))

    console.print(table)
    console.rule("[bold blue]✅ Training Completed[/bold blue]")

    logger.info(f"Training completed. Accuracy={acc:.4f}, Loss={loss:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

# ---------------- Scheduler Loop ----------------
if __name__ == "__main__":
    interval_minutes = 2
    while True:
        now = datetime.now()
        console.print(f"\n⏰ Triggered at: {now.strftime('%Y-%m-%d %H:%M:%S')}", style="bold yellow")
        train_model()

        # Calculate next trigger time
        next_trigger = now + timedelta(minutes=interval_minutes)
        console.print(f"[bold yellow]⏳ Next run scheduled at: {next_trigger.strftime('%Y-%m-%d %H:%M:%S')}[/bold yellow]")
        time.sleep(interval_minutes * 60)