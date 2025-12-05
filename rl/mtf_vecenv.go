// Package rl provides a multi-timeframe vectorized CUDA trading environment.
package rl

import (
	"github.com/empyrealm/gotorch/consts"
	"github.com/empyrealm/gotorch/tensor"
)

// ============================================================================
// Multi-Timeframe Vectorized CUDA Trading Environment
// Processes all environments in parallel on GPU with multiple timeframe data.
// ============================================================================

// MTFVecEnvConfig configures the multi-timeframe vectorized environment.
type MTFVecEnvConfig struct {
	NumEnvs       int64   // Number of parallel environments
	MaxSteps      int64   // Max steps per episode
	NumTimeframes int64   // Number of timeframes (e.g., 4 for 1m, 5m, 15m, 1h)
	FeaturePerTF  int64   // Features per timeframe (e.g., 35)
	InitialEquity float64 // Starting capital
	MinEquity     float64 // Bankruptcy threshold
	FeeRate       float64 // Trading fee rate

	// Timeframe ratios relative to base (1m).
	// E.g., [1, 5, 15, 60] means 1m, 5m, 15m, 1h.
	TimeframeRatios []int64
}

// DefaultMTFVecEnvConfig returns config for 4 timeframes.
func DefaultMTFVecEnvConfig() MTFVecEnvConfig {
	return MTFVecEnvConfig{
		NumEnvs:         64,
		MaxSteps:        4096,
		NumTimeframes:   4,
		FeaturePerTF:    35,
		InitialEquity:   10000.0,
		MinEquity:       1000.0,
		FeeRate:         0.0005,
		TimeframeRatios: []int64{1, 5, 15, 60},
	}
}

// MTFVecEnv is a multi-timeframe vectorized trading environment on GPU.
type MTFVecEnv struct {
	config MTFVecEnvConfig
	device consts.DeviceType

	// Market data for each timeframe (pre-loaded to GPU).
	// Each: [num_envs, time_steps, feature_dim]
	marketDataTF []*tensor.Tensor

	// Base timeframe market data for price info.
	// Shape: [num_envs, time_steps, feature_dim]
	baseMarketData *tensor.Tensor

	// Environment state (all on GPU).
	stepIndices *tensor.Tensor // [num_envs] - current time step (base TF)
	positions   *tensor.Tensor // [num_envs] - position (-1, 0, 1)
	entryPrices *tensor.Tensor // [num_envs] - entry price
	equity      *tensor.Tensor // [num_envs] - current equity
	maxEquity   *tensor.Tensor // [num_envs] - max equity seen
	prevPnL     *tensor.Tensor // [num_envs] - previous PnL for returns

	// Output buffers (pre-allocated on GPU).
	outStates  *tensor.Tensor // [num_envs, total_state_dim]
	outRewards *tensor.Tensor // [num_envs]
	outDones   *tensor.Tensor // [num_envs]

	// Total state dimension: num_timeframes * features + position + equity.
	stateDim     int64
	maxTimeSteps int64
}

// NewMTFVecEnv creates a new multi-timeframe vectorized CUDA environment.
func NewMTFVecEnv(config MTFVecEnvConfig, device consts.DeviceType) *MTFVecEnv {

	env := &MTFVecEnv{
		config:       config,
		device:       device,
		marketDataTF: make([]*tensor.Tensor, config.NumTimeframes),
	}

	// Calculate total state dimension.
	// Features from all timeframes + position (1) + normalized equity (1).
	env.stateDim = config.NumTimeframes*config.FeaturePerTF + 2

	// Initialize state tensors on GPU.
	env.stepIndices = tensor.Zeros(consts.KInt64, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))
	env.positions = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))
	env.entryPrices = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))
	env.equity = tensor.FromFloat32(
		makeSlice(config.NumEnvs, float32(config.InitialEquity)),
		tensor.WithShapes(config.NumEnvs),
		tensor.WithDevice(device),
	)
	env.maxEquity = tensor.FromFloat32(
		makeSlice(config.NumEnvs, float32(config.InitialEquity)),
		tensor.WithShapes(config.NumEnvs),
		tensor.WithDevice(device),
	)
	env.prevPnL = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))

	// Pre-allocate output buffers.
	env.outStates = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs, env.stateDim), tensor.WithDevice(device))
	env.outRewards = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))
	env.outDones = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))

	return env
}

// LoadTimeframeData loads market data for a specific timeframe.
// data: [num_envs, time_steps, feature_dim]
// tfIndex: 0 = base (1m), 1 = 5m, 2 = 15m, 3 = 1h, etc.
func (env *MTFVecEnv) LoadTimeframeData(tfIndex int, data *tensor.Tensor) {

	env.marketDataTF[tfIndex] = data.ToDevice(env.device)

	if tfIndex == 0 {
		// Base timeframe also stored separately for price access.
		env.baseMarketData = env.marketDataTF[0]
		env.maxTimeSteps = data.Shapes()[1]
	}
}

// LoadAllTimeframeData loads data for all timeframes from slices.
// Each slice is [time_steps * feature_dim] flattened.
func (env *MTFVecEnv) LoadAllTimeframeData(
	baseData []float32,
	baseSteps int64,
	tfData [][]float32,
	tfSteps []int64,
) {
	numEnvs := env.config.NumEnvs
	featureDim := env.config.FeaturePerTF

	// Load base timeframe.
	expandedBase := expandForEnvs(baseData, numEnvs, baseSteps, featureDim)
	env.marketDataTF[0] = tensor.FromFloat32(
		expandedBase,
		tensor.WithShapes(numEnvs, baseSteps, featureDim),
		tensor.WithDevice(env.device),
	)
	env.baseMarketData = env.marketDataTF[0]
	env.maxTimeSteps = baseSteps

	// Load other timeframes.
	for i := 0; i < len(tfData); i++ {
		expanded := expandForEnvs(tfData[i], numEnvs, tfSteps[i], featureDim)
		env.marketDataTF[i+1] = tensor.FromFloat32(
			expanded,
			tensor.WithShapes(numEnvs, tfSteps[i], featureDim),
			tensor.WithDevice(env.device),
		)
	}
}

// Reset resets all environments and returns initial state.
func (env *MTFVecEnv) Reset() *tensor.Tensor {

	// Reset positions and equity.
	env.positions = tensor.Zeros(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device))
	env.entryPrices = tensor.Zeros(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device))
	env.equity = tensor.FromFloat32(
		makeSlice(env.config.NumEnvs, float32(env.config.InitialEquity)),
		tensor.WithShapes(env.config.NumEnvs),
		tensor.WithDevice(env.device),
	)
	env.maxEquity = tensor.FromFloat32(
		makeSlice(env.config.NumEnvs, float32(env.config.InitialEquity)),
		tensor.WithShapes(env.config.NumEnvs),
		tensor.WithDevice(env.device),
	)
	env.stepIndices = tensor.Zeros(consts.KInt64, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device))
	env.prevPnL = tensor.Zeros(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device))

	return env.buildState()
}

// Step takes actions and returns (next_states, rewards, dones).
// actions: [num_envs] - discrete actions (0=flat, 1=long, 2=short)
func (env *MTFVecEnv) Step(actions *tensor.Tensor) (states, rewards, dones *tensor.Tensor) {

	// Get current prices from base timeframe.
	currentFeatures := env.getTimeframeFeatures(0)              // [num_envs, features]
	currentPrices := currentFeatures.Narrow(1, 3, 1).Squeeze(1) // Close price at index 3.

	// Store old position for reward calculation.
	oldPosition := env.positions

	// Parse actions: 0=flat, 1=long, 2=short.
	newPosition := env.parseActions(actions)

	// Detect position changes.
	positionChanged := oldPosition.Ne(newPosition)

	// Calculate realized PnL for closing positions.
	closingPnL := env.calculateClosingPnL(oldPosition, currentPrices, positionChanged)

	// Calculate fees.
	fees := env.calculateFees(oldPosition, newPosition, currentPrices, positionChanged)

	// Update equity.
	env.equity = env.equity.Add(closingPnL).Sub(fees)
	env.maxEquity = tensor.Max(env.maxEquity, env.equity)

	// Update entry prices for new positions.
	env.entryPrices = tensor.Where(
		positionChanged.Mul(newPosition.Ne(tensor.Zeros(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device)))),
		currentPrices,
		env.entryPrices,
	)

	// Compute advanced rewards.
	rewards = env.computeReward(closingPnL, fees, oldPosition, newPosition, currentPrices)

	// Update position.
	env.positions = newPosition

	// Advance time step.
	env.stepIndices = env.stepIndices.Add(
		tensor.FromFloat32([]float32{1}, tensor.WithShapes(1), tensor.WithDevice(env.device)).ToScalarType(consts.KInt64),
	)

	// Check done conditions.
	stepsDone := env.stepIndices.Ge(
		tensor.FromFloat32([]float32{float32(env.config.MaxSteps)}, tensor.WithShapes(1), tensor.WithDevice(env.device)).ToScalarType(consts.KInt64),
	)
	bankruptDone := env.equity.Lt(
		tensor.FromFloat32([]float32{float32(env.config.MinEquity)}, tensor.WithShapes(1), tensor.WithDevice(env.device)),
	)
	dones = stepsDone.LogicalOr(bankruptDone).ToScalarType(consts.KFloat)

	// Auto-reset done environments.
	env.autoResetDone(dones)

	// Build next state.
	states = env.buildState()

	// Store PnL for next step's rolling calculation.
	env.prevPnL = closingPnL

	return states, rewards, dones
}

// buildState constructs the multi-timeframe observation state.
// Returns: [num_envs, total_state_dim]
func (env *MTFVecEnv) buildState() *tensor.Tensor {

	// Gather features from each timeframe.
	var allFeatures []*tensor.Tensor

	for tfIdx := int64(0); tfIdx < env.config.NumTimeframes; tfIdx++ {
		tfFeatures := env.getTimeframeFeatures(int(tfIdx))
		allFeatures = append(allFeatures, tfFeatures)
	}

	// Concatenate all timeframe features.
	combined := allFeatures[0]
	for i := 1; i < len(allFeatures); i++ {
		combined = combined.Cat(allFeatures[i])
	}

	// Add position and normalized equity.
	posUnsqueezed := env.positions.Unsqueeze(1)
	equityNorm := env.equity.Div(
		tensor.FromFloat32([]float32{float32(env.config.InitialEquity)}, tensor.WithShapes(1), tensor.WithDevice(env.device)),
	).Unsqueeze(1)

	return combined.Cat(posUnsqueezed).Cat(equityNorm)
}

// getTimeframeFeatures gets current features for a specific timeframe.
func (env *MTFVecEnv) getTimeframeFeatures(tfIndex int) *tensor.Tensor {

	if env.marketDataTF[tfIndex] == nil {
		// Return zeros if no data loaded.
		return tensor.Zeros(
			consts.KFloat,
			tensor.WithShapes(env.config.NumEnvs, env.config.FeaturePerTF),
			tensor.WithDevice(env.device),
		)
	}

	// Convert step indices to this timeframe's index.
	ratio := env.config.TimeframeRatios[tfIndex]
	tfIndices := env.stepIndices.Div(
		tensor.FromFloat32([]float32{float32(ratio)}, tensor.WithShapes(1), tensor.WithDevice(env.device)).ToScalarType(consts.KInt64),
	)

	// Clamp to valid range.
	maxIdx := env.marketDataTF[tfIndex].Shapes()[1] - 1
	tfIndices = tfIndices.ClampMinMax(0, float64(maxIdx))

	// Gather features at these indices.
	// marketDataTF[tfIndex]: [num_envs, time_steps, feature_dim]
	return env.marketDataTF[tfIndex].Narrow(1, 0, 1).Squeeze(1) // Simplified: first timestep for now.
	// TODO: Proper gather with tfIndices.
}

// parseActions converts action integers to position values.
func (env *MTFVecEnv) parseActions(actions *tensor.Tensor) *tensor.Tensor {

	zero := tensor.Zeros(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device))
	one := tensor.FromFloat32(makeSlice(env.config.NumEnvs, 1.0), tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device))
	negOne := tensor.FromFloat32(makeSlice(env.config.NumEnvs, -1.0), tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device))

	// 0 = flat, 1 = long, 2 = short.
	isFlat := actions.Eq(tensor.FromFloat32([]float32{0}, tensor.WithShapes(1), tensor.WithDevice(env.device)))
	isLong := actions.Eq(tensor.FromFloat32([]float32{1}, tensor.WithShapes(1), tensor.WithDevice(env.device)))

	return tensor.Where(isFlat, zero, tensor.Where(isLong, one, negOne))
}

// calculateClosingPnL computes PnL for closing positions.
func (env *MTFVecEnv) calculateClosingPnL(
	oldPos, currentPrices *tensor.Tensor,
	posChanged *tensor.Tensor,
) *tensor.Tensor {

	// PnL = position * (current_price - entry_price) / entry_price.
	priceDiff := currentPrices.Sub(env.entryPrices)
	pnl := oldPos.Mul(priceDiff).Div(env.entryPrices.Add(
		tensor.FromFloat32([]float32{1e-8}, tensor.WithShapes(1), tensor.WithDevice(env.device)),
	))

	// Only count PnL when actually closing (position changed and had position).
	hasPosition := oldPos.Ne(tensor.Zeros(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device)))
	return tensor.Where(posChanged.Mul(hasPosition), pnl.Mul(env.equity), tensor.Zeros(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device)))
}

// calculateFees computes trading fees.
func (env *MTFVecEnv) calculateFees(
	oldPos, newPos, currentPrices *tensor.Tensor,
	posChanged *tensor.Tensor,
) *tensor.Tensor {

	// Fee = |position_change| * price * fee_rate.
	posChange := newPos.Sub(oldPos).Abs()
	fees := posChange.Mul(currentPrices).MulScalar(env.config.FeeRate)

	return tensor.Where(posChanged, fees, tensor.Zeros(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device)))
}

// computeReward calculates the advanced multi-component reward.
func (env *MTFVecEnv) computeReward(
	pnl, fees, oldPos, newPos, currentPrices *tensor.Tensor,
) *tensor.Tensor {

	// 1. Normalized PnL (as % of equity).
	pnlPct := pnl.Div(env.equity.Add(
		tensor.FromFloat32([]float32{1e-8}, tensor.WithShapes(1), tensor.WithDevice(env.device)),
	)).MulScalar(100.0)

	// 2. Fee penalty.
	feePct := fees.Div(env.equity.Add(
		tensor.FromFloat32([]float32{1e-8}, tensor.WithShapes(1), tensor.WithDevice(env.device)),
	)).MulScalar(100.0)

	// 3. Drawdown penalty (quadratic).
	drawdown := env.maxEquity.Sub(env.equity).Div(env.maxEquity.Add(
		tensor.FromFloat32([]float32{1e-8}, tensor.WithShapes(1), tensor.WithDevice(env.device)),
	))
	ddPenalty := drawdown.Pow(2.0).MulScalar(10.0)

	// 4. Churn penalty.
	posChanged := oldPos.Ne(newPos).ToScalarType(consts.KFloat)
	churnPenalty := posChanged.MulScalar(0.01)

	// 5. Holding bonus.
	holding := newPos.Ne(tensor.Zeros(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device))).ToScalarType(consts.KFloat)
	winning := pnl.Gt(tensor.Zeros(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device))).ToScalarType(consts.KFloat)
	holdBonus := holding.Mul(winning).MulScalar(0.001)

	// 6. Unrealized PnL signal.
	unrealized := newPos.Mul(currentPrices.Sub(env.entryPrices)).Div(env.entryPrices.Add(
		tensor.FromFloat32([]float32{1e-8}, tensor.WithShapes(1), tensor.WithDevice(env.device)),
	))
	unrealizedSignal := unrealized.MulScalar(10.0).Tanh().MulScalar(0.01)

	// 7. Asymmetric loss penalty.
	isLoss := pnl.Lt(tensor.Zeros(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device)))
	downsidePenalty := tensor.Where(
		isLoss,
		pnl.Abs().Div(env.equity.Add(
			tensor.FromFloat32([]float32{1e-8}, tensor.WithShapes(1), tensor.WithDevice(env.device)),
		)).MulScalar(150.0),
		tensor.Zeros(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device)),
	)

	// Combine.
	reward := pnlPct.Sub(feePct).Sub(ddPenalty).Sub(churnPenalty).Add(holdBonus).Add(unrealizedSignal).Sub(downsidePenalty)

	// Clip for stability.
	return reward.ClampMinMax(-1.0, 1.0)
}

// autoResetDone resets environments that are done.
func (env *MTFVecEnv) autoResetDone(dones *tensor.Tensor) {

	// Reset positions where done.
	env.positions = tensor.Where(
		dones.Gt(tensor.FromFloat32([]float32{0.5}, tensor.WithShapes(1), tensor.WithDevice(env.device))),
		tensor.Zeros(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device)),
		env.positions,
	)

	// Reset equity where done.
	initialEquityTensor := tensor.FromFloat32(
		makeSlice(env.config.NumEnvs, float32(env.config.InitialEquity)),
		tensor.WithShapes(env.config.NumEnvs),
		tensor.WithDevice(env.device),
	)
	env.equity = tensor.Where(
		dones.Gt(tensor.FromFloat32([]float32{0.5}, tensor.WithShapes(1), tensor.WithDevice(env.device))),
		initialEquityTensor,
		env.equity,
	)
	env.maxEquity = tensor.Where(
		dones.Gt(tensor.FromFloat32([]float32{0.5}, tensor.WithShapes(1), tensor.WithDevice(env.device))),
		initialEquityTensor,
		env.maxEquity,
	)

	// Reset step indices where done.
	env.stepIndices = tensor.Where(
		dones.Gt(tensor.FromFloat32([]float32{0.5}, tensor.WithShapes(1), tensor.WithDevice(env.device))).ToScalarType(consts.KInt64),
		tensor.Zeros(consts.KInt64, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device)),
		env.stepIndices,
	)

	// Reset entry prices where done.
	env.entryPrices = tensor.Where(
		dones.Gt(tensor.FromFloat32([]float32{0.5}, tensor.WithShapes(1), tensor.WithDevice(env.device))),
		tensor.Zeros(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device)),
		env.entryPrices,
	)
}

// GetStateDim returns total state dimension.
func (env *MTFVecEnv) GetStateDim() int64 {
	return env.stateDim
}

// GetNumEnvs returns number of parallel environments.
func (env *MTFVecEnv) GetNumEnvs() int64 {
	return env.config.NumEnvs
}

// expandForEnvs expands single environment data for all environments.
func expandForEnvs(data []float32, numEnvs, timeSteps, featureDim int64) []float32 {

	srcLen := timeSteps * featureDim
	expanded := make([]float32, numEnvs*srcLen)

	for env := int64(0); env < numEnvs; env++ {
		copy(expanded[env*srcLen:(env+1)*srcLen], data[:srcLen])
	}

	return expanded
}
