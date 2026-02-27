import ray
import numpy as np
import logging
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.logger import DEFAULT_LOGGERS

# Configure logging level
logging.getLogger("ray.tune").setLevel(logging.WARNING)  # Reduce Ray Tune logs
logging.getLogger("ray.rllib").setLevel(logging.WARNING)  # Reduce RLLib logs

def train_fn(config):
    """Training function with smart debugging"""
    
    # Get trial info for better debugging
    trial_id = tune.get_trial_id()
    trial_name = tune.get_trial_name()
    
    print(f"\n=== Starting Trial: {trial_name} ===")
    print(f"Config: lr={config['lr']:.6f}, batch_size={config['batch_size']}, optimizer={config['optimizer']}")
    
    best_accuracy = 0
    for step in range(100):
        # Simulate training - better configs learn faster
        accuracy = (step / 100) * config["lr"] * 10
        
        # Add some randomness
        accuracy += np.random.normal(0, 0.1)
        
        # Track best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
        # Smart logging: more frequent at start, less frequent later
        should_log = (
            step < 10 or                    # First 10 steps
            step % 20 == 0 or              # Every 20 steps
            step == 99                     # Final step
        )
        
        if should_log:
            print(f"  Step {step:2d}: acc={accuracy:.4f} (best={best_accuracy:.4f})")
        
        # Report to Ray Tune
        tune.report({
            "accuracy": accuracy, 
            "step": step,
            "best_accuracy": best_accuracy
        })
    
    print(f"=== Trial {trial_name} finished with best accuracy: {best_accuracy:.4f} ===\n")

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

# Custom progress reporter for cleaner output
reporter = tune.CLIReporter(
    max_progress_rows=8,                    # Show only top 8 trials
    max_report_frequency=10,                # Update every 10 seconds
    print_intermediate_tables=True,
    metric_columns=["accuracy", "best_accuracy", "step"],
    parameter_columns=["lr", "batch_size", "optimizer"],
    sort_by_metric=True
)

print("🚀 Starting Ray Tune Experiment with Smart Debugging")
print(f"📊 Running {20} trials with ASHA scheduler")
print("=" * 60)

# Initialize Ray with reduced logging
ray.init(
    log_to_driver=False,        # Don't log worker outputs to driver
    logging_level=logging.ERROR # Only show errors
)

# Run with scheduler
analysis = tune.run(
    train_fn,
    config=config,
    num_samples=20,              # Start 20 trials
    scheduler=scheduler,
    metric="accuracy",
    mode="max",
    resources_per_trial={"cpu": 1},
    verbose=1,                   # Minimal verbosity
    progress_reporter=reporter,
    local_dir="./ray_results",   # Save results locally
    name="smart_debug_experiment"
)

# Print summary
print("\n🏆 EXPERIMENT COMPLETE!")
print("=" * 60)
best_trial = analysis.get_best_trial("accuracy", "max", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final accuracy: {best_trial.last_result['accuracy']:.4f}")

# Show top 5 trials
df = analysis.results_df
top_5 = df.nlargest(5, 'accuracy')[['config/lr', 'config/batch_size', 'config/optimizer', 'accuracy']]
print(f"\nTop 5 Trials:")
print(top_5.to_string(index=False))