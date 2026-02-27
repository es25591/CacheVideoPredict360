import ray
import numpy as np
import logging
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Silence Ray logging
logging.getLogger("ray").setLevel(logging.ERROR)
ray.init(log_to_driver=False, logging_level=logging.ERROR)

def train_fn(config):
    """Super quiet training function"""
    for step in range(100):
        accuracy = (step / 100) * config["lr"] * 10 + np.random.normal(0, 0.1)
        tune.report({"accuracy": accuracy})

# Simple experiment with minimal output
analysis = tune.run(
    train_fn,
    config={
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "optimizer": tune.choice(["adam", "sgd"])
    },
    num_samples=5,                           # Fewer trials for testing
    scheduler=ASHAScheduler(max_t=100, grace_period=10, reduction_factor=3),
    metric="accuracy",
    mode="max",
    verbose=0,                               # Completely silent
    progress_reporter=tune.CLIReporter(max_progress_rows=3, print_intermediate_tables=False)
)

# Just show the results
best = analysis.get_best_trial("accuracy", "max", "last")
print(f"✅ Best: lr={best.config['lr']:.6f}, accuracy={best.last_result['accuracy']:.4f}")