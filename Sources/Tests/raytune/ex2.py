import ray
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_fn(config):
    # Simulate training - better configs learn faster
    for step in range(100):
        accuracy = (step / 100) * config["lr"] * 10
        
        # Add some randomness
        accuracy += np.random.normal(0, 0.1)
        
        # Only print debug info every 10 steps
        if step % 10 == 0:
            print(f"Step {step}: accuracy={accuracy:.4f}, lr={config['lr']:.4f}")
        
        tune.report({"accuracy": accuracy, "step": step})

# Define search space
config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64, 128]),
    "optimizer": tune.choice(["adam", "sgd"])
}

# Configure ASHA scheduler
scheduler = ASHAScheduler(
    max_t=100,           # Max 100 steps per trial
    grace_period=10,     # Don't stop before 10 steps
    reduction_factor=3,  # Keep top 1/3 of trials each round
)

# Run with scheduler
analysis = tune.run(
    train_fn,
    config=config,
    num_samples=20,      # Start 20 trials
    scheduler=scheduler,
    metric="accuracy",
    mode="max",
    resources_per_trial={"cpu": 1},
    verbose=1,           # Control verbosity: 0=silent, 1=minimal, 2=normal, 3=verbose
    progress_reporter=tune.CLIReporter(
        max_progress_rows=10,           # Limit displayed trials
        max_report_frequency=30,        # Update every 30 seconds
        print_intermediate_tables=True,
        metric_columns=["accuracy", "step", "training_iteration"]
    )
)