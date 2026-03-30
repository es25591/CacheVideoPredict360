# A2C Stability Implementation Summary

## Overview
Implemented GAE (Generalized Advantage Estimation) with entropy regularization and advantage normalization to stabilize A2C training with large n-step (1000) for tile cache prediction.

## Changes Made

### 1. **Configuration Updates** (`Sources/Configs/config.py`)
Added new hyperparameters:
```python
gae_lambda: float = 0.95              # GAE lambda for variance reduction
entropy_beta: float = 0.01            # Entropy regularization coefficient (0.01-0.05)
advantage_clip: float = 5.0           # Clip advantages to [-5, 5] range
gradient_clip_norm: float = 0.5       # Max gradient norm clipping

# Updated learning rates
learning_rate_actor: float = 1e-3
learning_rate_critic: float = 1e-3

# Increased batch size for more stable estimates
batch_size: int = 64  # was 32
```

### 2. **Buffer Enhancement** (`Sources/RL/Buffers.py`)
Added `compute_gae()` method to `NStepReplayBuffer`:
```python
def compute_gae(self, rewards, values, next_value, dones, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    Smooths high-variance n-step returns while preserving temporal information.
    
    Returns:
        advantages: GAE advantages smoothed across temporal horizon
        returns: TD targets for critic (advantage + value)
    """
```

**Key Formula:**
- TD Residual: δ(t) = r(t) + γV(s_{t+1}) - V(s_t)
- GAE: Â(t) = δ(t) + (λγ)δ(t+1) + (λγ)²δ(t+2) + ...
- Returns: R(t) = Â(t) + V(s_t)

### 3. **A2CWorker Core Changes** (`Sources/RL/A2CWorker.py`)

#### Hyperparameter Registration (\_\_init__)
```python
self.gae_lambda = cfg.gae_lambda
self.entropy_beta = cfg.entropy_beta
self.advantage_clip = cfg.advantage_clip
self.gradient_clip_norm = cfg.gradient_clip_norm
```

#### Learn Method Improvements
1. **GAE Computation**
   - Replaced raw n-step returns with GAE advantages
   - Reduces variance while maintaining n-step temporal awareness for tile popularity dynamics

2. **Advantage Normalization**
   ```python
   advantage_mean = advantages.mean()
   advantage_std = advantages.std() + 1e-8
   advantages = (advantages - advantage_mean) / advantage_std
   advantages = torch.clamp(advantages, -self.advantage_clip, self.advantage_clip)
   ```
   - Handles PSNR reward scale (0-40) properly
   - Prevents extreme gradient updates

3. **Entropy Regularization**
   ```python
   entropy_loss = -self.entropy_beta * entropy.mean()
   actor_total_loss = actor_loss + entropy_loss
   ```
   - Encourages exploration across actions (preventing policy collapse)
   - Trade-off between exploitation and exploration

4. **Enhanced Metrics Tracking**
   ```python
   metrics = {
       'train_loss': float(total_loss.item()),
       'actor_loss': float(actor_loss.item()),
       'entropy_loss': float(entropy_loss.item()),
       'critic_loss': float(critic_loss.item()),
       'policy_entropy': float(entropy.mean().item()),  # NEW
       'advantage_mean': float(advantage_mean.item()),   # NEW
       'advantage_std': float(advantage_std.item()),     # NEW
       'advantage_min': float(advantages.min().item()),  # NEW
       'advantage_max': float(advantages.max().item())   # NEW
   }
   ```

## Expected Improvements

### Stability Enhancements
- **Variance Reduction**: GAE smooths advantage estimates by interpolating between 1-step and n-step returns
- **Entropy Exploration**: Prevents policy convergence to single action prematurely
- **Gradient Control**: Advantage clipping + normalization prevents extreme updates

### Training Behavior
1. **More Stable Convergence**: Reward curves should show less variance between episodes
2. **Better Credit Assignment**: Tile popularity changes over time properly reflected in actions
3. **Improved Policy Logic**: Actions should correlate with predicted viewport and cache state

### Metrics to Monitor
```
✓ policy_entropy    → Should stay > 0.1 (policy not collapsed)
✓ advantage_std     → Should be ~ 1.0 (normalized correctly)
✓ actor_loss        → Should decrease gradually (not spike)
✓ critic_loss       → Should converge smoothly
✓ cache_hit_rate    → Primary performance metric (should improve)
```

## Tuning Recommendations

### If Training is Still Unstable:
1. **Increase GAE lambda** (0.95 → 0.99)
   - Closer to Monte Carlo returns, more temporal information
   - Trade-off: higher variance

2. **Adjust entropy_beta** (start: 0.01)
   - Too low (0.001): Policy collapses to few actions
   - Too high (0.05+): Actions become random
   - Monitor policy_entropy in logs

3. **Reduce n_step incrementally** (1000 → 500 → 100)
   - Only if GAE + entropy don't solve instability
   - Start with n_step=500 to test

### If Cache Hits Not Improving:
1. **Check advantage statistics**
   - advantage_std should be ~1.0 (normalized)
   - advantage_min/max should be within [-5, 5] (clipped)
   - If not, tuning issue (learning rate too high?)

2. **Verify tile popularity is captured**
   - Log actor decision patterns when popularity changes
   - Create visualization of tile selection over episodes
   - Check if actions correlate with predicted viewport

3. **Reward Scaling**
   - PSNR scale (0-40) is already normalized
   - If cache hits are very sparse, consider reward shaping:
     - Add small penalty for cache misses: `reward = psnr - miss_penalty`
     - Encourage early caching: `reward += time_to_use_bonus`

## Testing Procedure

### Phase 1: Smoke Test (Quick Validation)
```bash
# Just verify code loads without errors
python -c "from Sources.RL.A2CWorker import A2CWorker; print('✓ Code loads')"
```

### Phase 2: Quick Training (100-200 episodes)
```python
# Run with reduced n_step first (n_step=100)
# Monitor: Does training start? Are metrics reasonable?
# Target: Smooth metric curves, no NaN/Inf
```

### Phase 3: Full Convergence Test (1000+ episodes)
```
Track:
- Episode reward trajectory (should improve)
- Cache hit rate (main metric)
- Policy entropy (should stay > 0.1)
- Advantage statistics (std ~ 1.0)
```

### Phase 4: Comparison vs DQN
```
Run both A2C (new) and DQN on same dataset/seeds
Compare:
- Final cache hit rates
- Convergence speed (steps to reasonable performance)
- Training stability (variance of reward curves)
- Computational cost
```

## Hyperparameter Tuning Grid

If performance isn't meeting targets, try:

| n_step | gae_lambda | entropy_beta | Result Expectation |
|--------|-----------|--------------|-------------------|
| 1000   | 0.95      | 0.01         | **Start here** (current) |
| 1000   | 0.99      | 0.01         | More temporal info, higher variance |
| 500    | 0.95      | 0.01         | Faster updates, less bias |
| 1000   | 0.95      | 0.05         | Strong exploration (might be too random) |
| 1000   | 0.95      | 0.005        | Weak exploration (might collapse) |

## Implementation Quality Checklist

- ✅ GAE computation mathematically correct
- ✅ Advantage normalization per-batch
- ✅ Entropy regularization on actor loss
- ✅ Gradient clipping on both networks
- ✅ Enhanced metrics for debugging
- ✅ Batch size increased to 64 for stability
- ✅ Separate actor/critic learning rates configured
- ✅ Handles PSNR reward scale (0-40) properly

## References & Mathematical Details

### GAE (Generalized Advantage Estimation)
Paper: High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2015)

**Why GAE helps with n-step:**
- n-step returns have high variance with large n
- GAE interpolates between small-step (high bias, low variance) and large n-step (low bias, high variance)
- λ parameter controls trade-off: λ=0 → 1-step TD, λ=1 → Monte Carlo

### Entropy Regularization
- Policy entropy: H(π) = -Σ π(a|s) log π(a|s)
- Encourages exploration: policy_gradient + β * H(π)
- Prevents deterministic policy collapse

### Advantage Normalization
- Standardizes advantage scale across batches
- Critical when reward scale is heterogeneous (PSNR: 0-40)
- Reduces sensitivity to outliers in estimation

## Next Steps

1. **Run initial training** with current config
2. **Monitor metrics** (esp. policy_entropy and advantage_std)
3. **Compare cache hit rates** vs baseline DQN
4. **Tune if needed** using grid above
5. **Document final hyperparams** and performance results

---

**Implementation Date**: 2026-03-30  
**Based on**: Discussion of tile popularity temporal dynamics in 360° cache prediction
