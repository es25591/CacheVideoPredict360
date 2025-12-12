import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files
fov_predictor = pd.read_csv('fov_predictor_500MB.csv')
probs_predictor = pd.read_csv('fov_predictor.csv')

# Calculate summary statistics
print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\nFoV Predictor:")
print(f"  Average Cache Hits: {fov_predictor['cache_hits'].mean():.2f}")
print(f"  Average Cache Misses: {fov_predictor['cache_misses'].mean():.2f}")
print(f"  Total Cache Hits: {fov_predictor['cache_hits'].sum()}")
print(f"  Total Cache Misses: {fov_predictor['cache_misses'].sum()}")
print(f"  Hit Rate: {fov_predictor['cache_hits'].sum() / (fov_predictor['cache_hits'].sum() + fov_predictor['cache_misses'].sum()) * 100:.2f}%")
print(f"  Average Total Reward: {fov_predictor['total_reward'].mean():.2f}")

print("\nProbs Predictor:")
print(f"  Average Cache Hits: {probs_predictor['cache_hits'].mean():.2f}")
print(f"  Average Cache Misses: {probs_predictor['cache_misses'].mean():.2f}")
print(f"  Total Cache Hits: {probs_predictor['cache_hits'].sum()}")
print(f"  Total Cache Misses: {probs_predictor['cache_misses'].sum()}")
print(f"  Hit Rate: {probs_predictor['cache_hits'].sum() / (probs_predictor['cache_hits'].sum() + probs_predictor['cache_misses'].sum()) * 100:.2f}%")
print(f"  Average Total Reward: {probs_predictor['total_reward'].mean():.2f}")

print("\nImprovement (Probs over FoV):")
hits_improvement = ((probs_predictor['cache_hits'].mean() - fov_predictor['cache_hits'].mean()) / fov_predictor['cache_hits'].mean()) * 100
misses_improvement = ((fov_predictor['cache_misses'].mean() - probs_predictor['cache_misses'].mean()) / fov_predictor['cache_misses'].mean()) * 100
reward_improvement = ((probs_predictor['total_reward'].mean() - fov_predictor['total_reward'].mean()) / fov_predictor['total_reward'].mean()) * 100

print(f"  Cache Hits: {hits_improvement:+.2f}%")
print(f"  Cache Misses Reduction: {misses_improvement:+.2f}%")
print(f"  Total Reward: {reward_improvement:+.2f}%")

# Calculate hit rates per episode
fov_predictor['hit_rate'] = fov_predictor['cache_hits'] / (fov_predictor['cache_hits'] + fov_predictor['cache_misses']) * 100
probs_predictor['hit_rate'] = probs_predictor['cache_hits'] / (probs_predictor['cache_hits'] + probs_predictor['cache_misses']) * 100

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

episodes = range(1, len(fov_predictor) + 1)

# 1. Cache Hits Comparison
axes[0, 0].plot(episodes, fov_predictor['cache_hits'], label='FoV Predictor', alpha=0.7, color='blue')
axes[0, 0].plot(episodes, probs_predictor['cache_hits'], label='Probs Predictor', alpha=0.7, color='green')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Cache Hits')
axes[0, 0].set_title('Cache Hits Comparison')
axes[0, 0].set_yscale('log')
axes[0, 0].set_ylim(bottom=0.1)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# 2. Cache Misses Comparison
axes[0, 1].plot(episodes, fov_predictor['cache_misses'], label='FoV Predictor', alpha=0.7, color='blue')
axes[0, 1].plot(episodes, probs_predictor['cache_misses'], label='Probs Predictor', alpha=0.7, color='green')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Cache Misses')
axes[0, 1].set_title('Cache Misses Comparison')
axes[0, 1].set_yscale('log')
axes[0, 1].set_ylim(bottom=0.1)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# 3. Hit Rate Comparison
axes[0, 2].plot(episodes, fov_predictor['hit_rate'], label='FoV Predictor', alpha=0.7, color='blue')
axes[0, 2].plot(episodes, probs_predictor['hit_rate'], label='Probs Predictor', alpha=0.7, color='green')
axes[0, 2].set_xlabel('Episode')
axes[0, 2].set_ylabel('Hit Rate (%)')
axes[0, 2].set_title('Cache Hit Rate Comparison')
axes[0, 2].set_yscale('log')
axes[0, 2].set_ylim(bottom=0.1)
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend()

# 4. Total Reward Comparison
axes[1, 0].plot(episodes, fov_predictor['total_reward'], label='FoV Predictor', alpha=0.7, color='blue')
axes[1, 0].plot(episodes, probs_predictor['total_reward'], label='Probs Predictor', alpha=0.7, color='green')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Total Reward')
axes[1, 0].set_title('Total Reward Comparison')
axes[1, 0].set_yscale('log')
axes[1, 0].set_ylim(bottom=0.1)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# 5. Moving Average of Cache Hits
w = 20
if w > 1:
    fov_ma_hits = [sum(fov_predictor['cache_hits'].iloc[i - w:i]) / w for i in range(w, len(fov_predictor) + 1)]
    probs_ma_hits = [sum(probs_predictor['cache_hits'].iloc[i - w:i]) / w for i in range(w, len(probs_predictor) + 1)]
    axes[1, 1].plot(range(w + 1, len(fov_predictor) + 2), fov_ma_hits, label='FoV Predictor', color='blue')
    axes[1, 1].plot(range(w + 1, len(probs_predictor) + 2), probs_ma_hits, label='Probs Predictor', color='green')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Cache Hits (Moving Avg)')
axes[1, 1].set_title(f'Cache Hits Moving Average (w={w})')
axes[1, 1].set_yscale('log')
axes[1, 1].set_ylim(bottom=0.1)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

# 6. Moving Average of Cache Misses
if w > 1:
    fov_ma_misses = [sum(fov_predictor['cache_misses'].iloc[i - w:i]) / w for i in range(w, len(fov_predictor) + 1)]
    probs_ma_misses = [sum(probs_predictor['cache_misses'].iloc[i - w:i]) / w for i in range(w, len(probs_predictor) + 1)]
    axes[1, 2].plot(range(w + 1, len(fov_predictor) + 2), fov_ma_misses, label='FoV Predictor', color='blue')
    axes[1, 2].plot(range(w + 1, len(probs_predictor) + 2), probs_ma_misses, label='Probs Predictor', color='green')
axes[1, 2].set_xlabel('Episode')
axes[1, 2].set_ylabel('Cache Misses (Moving Avg)')
axes[1, 2].set_title(f'Cache Misses Moving Average (w={w})')
axes[1, 2].set_yscale('log')
axes[1, 2].set_ylim(bottom=0.1)
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('predictor_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved as 'predictor_comparison.png'")
plt.show()

# Create a box plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Box plot for Cache Hits
axes[0].boxplot([fov_predictor['cache_hits'], probs_predictor['cache_hits']], 
                labels=['FoV Predictor', 'Probs Predictor'])
axes[0].set_ylabel('Cache Hits')
axes[0].set_title('Cache Hits Distribution')
axes[0].set_ylim(bottom=0)  # <--- Forces y-axis to start at 0
axes[0].grid(True, alpha=0.3, axis='y')

# Box plot for Cache Misses
axes[1].boxplot([fov_predictor['cache_misses'], probs_predictor['cache_misses']], 
                labels=['FoV Predictor', 'Probs Predictor'])
axes[1].set_ylabel('Cache Misses')
axes[1].set_title('Cache Misses Distribution')
axes[1].set_ylim(bottom=0)  # <--- Forces y-axis to start at 0
axes[1].grid(True, alpha=0.3, axis='y')

# Box plot for Total Reward
axes[2].boxplot([fov_predictor['total_reward'], probs_predictor['total_reward']], 
                labels=['FoV Predictor', 'Probs Predictor'])
axes[2].set_ylabel('Total Reward')
axes[2].set_title('Total Reward Distribution')
axes[2].set_ylim(bottom=0)  # <--- Forces y-axis to start at 0
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('predictor_comparison_boxplots.png', dpi=300, bbox_inches='tight')
print(f"Box plots saved as 'predictor_comparison_boxplots.png'")
plt.show()
