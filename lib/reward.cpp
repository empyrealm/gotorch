// reward.cpp
// Advanced reward computation for RL trading.
// Implements Sharpe ratio, risk-adjusted returns, and position management bonuses.

#include <torch/torch.h>
#include <cmath>


// ============================================================================
// Advanced Reward Computation (CUDA-accelerated)
// ============================================================================

// Compute rolling statistics for Sharpe ratio on GPU.
// returns: [batch, window_size] - recent returns
// Returns: mean, std for each batch element
std::pair<torch::Tensor, torch::Tensor> compute_rolling_stats(
    const torch::Tensor& returns,
    int64_t window_size
) {
    auto batch_size = returns.size(0);
    
    // Pad if not enough history.
    auto padded = returns;
    if (returns.size(1) < window_size) {
        auto padding = torch::zeros({batch_size, window_size - returns.size(1)}, returns.options());
        padded = torch::cat({padding, returns}, 1);
    }
    
    // Take last window_size elements.
    auto windowed = padded.index({torch::indexing::Slice(), torch::indexing::Slice(-window_size, torch::indexing::None)});
    
    auto mean = windowed.mean(1);
    auto std = windowed.std(1) + 1e-8;  // Prevent division by zero.
    
    return {mean, std};
}


// Advanced reward function with multiple components.
// All tensors are [batch_size].
torch::Tensor compute_advanced_reward(
    const torch::Tensor& pnl,           // Current step PnL
    const torch::Tensor& fees,          // Trading fees
    const torch::Tensor& equity,        // Current equity
    const torch::Tensor& max_equity,    // Max equity seen
    const torch::Tensor& position,      // Current position (-1, 0, 1)
    const torch::Tensor& old_position,  // Previous position
    const torch::Tensor& return_buffer, // Rolling returns buffer [batch, window]
    const torch::Tensor& volatility,    // Rolling volatility
    double risk_free_rate,              // Annualized risk-free rate
    double sharpe_weight,               // Weight for Sharpe component
    double drawdown_weight,             // Weight for drawdown penalty
    double churn_penalty,               // Penalty for excessive trading
    double hold_bonus                   // Bonus for holding positions
) {
    auto device = pnl.device();
    auto batch_size = pnl.size(0);
    
    // 1. Base PnL reward (normalized by equity).
    auto base_reward = pnl / (equity + 1e-8);
    
    // 2. Fee penalty.
    auto fee_penalty = fees / (equity + 1e-8);
    
    // 3. Drawdown penalty (exponential to heavily penalize large drawdowns).
    auto drawdown = (max_equity - equity) / (max_equity + 1e-8);
    auto drawdown_penalty = torch::pow(drawdown, 2.0) * drawdown_weight;
    
    // 4. Rolling Sharpe ratio bonus.
    // Compute rolling mean and std of returns.
    auto [roll_mean, roll_std] = compute_rolling_stats(return_buffer, 20);
    auto sharpe = (roll_mean - risk_free_rate / 252.0) / roll_std;  // Daily Sharpe
    auto sharpe_bonus = torch::clamp(sharpe, -2.0, 2.0) * sharpe_weight;
    
    // 5. Churn penalty - penalize frequent position changes.
    auto position_changed = (position != old_position).to(torch::kFloat32);
    auto churn_cost = position_changed * churn_penalty;
    
    // 6. Hold bonus - small reward for staying in profitable positions.
    auto holding = (position != 0).to(torch::kFloat32);
    auto profitable = (pnl > 0).to(torch::kFloat32);
    auto hold_reward = holding * profitable * hold_bonus;
    
    // 7. Risk-adjusted return (Sortino-like, only penalize downside).
    auto downside = torch::where(pnl < 0, pnl.pow(2), torch::zeros_like(pnl));
    auto downside_penalty = torch::sqrt(downside) * 0.5;
    
    // Combine all components.
    auto reward = base_reward
                - fee_penalty
                - drawdown_penalty
                + sharpe_bonus
                - churn_cost
                + hold_reward
                - downside_penalty;
    
    // Clip extreme rewards for stability.
    return torch::clamp(reward, -1.0, 1.0);
}


// Simplified version for single-step computation.
torch::Tensor compute_simple_reward(
    const torch::Tensor& pnl,
    const torch::Tensor& fees,
    const torch::Tensor& equity,
    const torch::Tensor& max_equity,
    const torch::Tensor& position,
    const torch::Tensor& old_position
) {
    // 1. Normalized PnL.
    auto base = pnl / (equity.abs() + 1e-8) * 100.0;  // Scale to percentage.
    
    // 2. Fee penalty.
    auto fee_pen = fees / (equity.abs() + 1e-8) * 100.0;
    
    // 3. Drawdown penalty (quadratic).
    auto drawdown = (max_equity - equity) / (max_equity + 1e-8);
    auto dd_pen = drawdown.pow(2) * 10.0;  // 10x weight on drawdown.
    
    // 4. Churn penalty.
    auto changed = (position != old_position).to(torch::kFloat32);
    auto churn = changed * 0.001;  // Small penalty per trade.
    
    // 5. Position holding bonus (encourages conviction).
    auto holding = (position != 0).to(torch::kFloat32);
    auto winning = (pnl > 0).to(torch::kFloat32);
    auto hold_bonus = holding * winning * 0.0001;
    
    // Combine.
    auto reward = base - fee_pen - dd_pen - churn + hold_bonus;
    
    // Clip for stability.
    return torch::clamp(reward, -1.0, 1.0);
}

