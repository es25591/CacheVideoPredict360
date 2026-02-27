import ray
from ray import tune

ray.init()

def train_fn(config):
    # Your training loop
    for step in range(10):
        acc = (step + 1) * config["lr"]  # fake metric
        tune.report({"mean_accuracy": acc})   # log metrics back to Tune

tune.run(
    train_fn,
    config={
        "lr": tune.grid_search([0.01, 0.1, 1.0]),  # search space
        "batch_size": tune.choice([16, 32, 64])
    },
    metric="mean_accuracy",   # the metric to optimize
    mode="max",               # whether to minimize/maximize
    num_samples=1             # number of times to sample configs
)
