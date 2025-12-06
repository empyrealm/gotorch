// Package rl provides a vectorized CUDA trading environment.
package rl

import (
	"github.com/empyrealm/gotorch/consts"
	"github.com/empyrealm/gotorch/internal/torch"
	"github.com/empyrealm/gotorch/tensor"
)

// ============================================================================
// Vectorized CUDA Trading Environment
// Processes all environments in parallel on GPU - no CPU transfer.
// ============================================================================

// VecEnvConfig configures the vectorized environment.
type VecEnvConfig struct {
	NumEnvs       int64   // Number of parallel environments
	MaxSteps      int64   // Max steps per episode
	FeatureDim    int64   // Market feature dimension
	InitialEquity float64 // Starting capital
	MinEquity     float64 // Bankruptcy threshold
	FeeRate       float64 // Trading fee rate (e.g., 0.001 = 0.1%)
}

// VecEnv is a vectorized trading environment running entirely on GPU.
type VecEnv struct {
	config VecEnvConfig
	device consts.DeviceType

	// Market data (pre-loaded to GPU).
	// Shape: [num_envs, time_steps, feature_dim]
	marketData *tensor.Tensor

	// Environment state (all on GPU).
	stepIndices *tensor.Tensor // [num_envs] - current time step
	positions   *tensor.Tensor // [num_envs] - position (-1, 0, 1)
	entryPrices *tensor.Tensor // [num_envs] - entry price
	equity      *tensor.Tensor // [num_envs] - current equity
	maxEquity   *tensor.Tensor // [num_envs] - max equity seen

	// Output buffers (pre-allocated on GPU).
	outStates      *tensor.Tensor
	outRewards     *tensor.Tensor
	outDones       *tensor.Tensor
	outPositions   *tensor.Tensor
	outEntryPrices *tensor.Tensor
	outEquity      *tensor.Tensor
	outMaxEquity   *tensor.Tensor
	outStepIndices *tensor.Tensor

	// Total time steps available.
	maxTimeSteps int64
}

// NewVecEnv creates a new vectorized CUDA trading environment.
func NewVecEnv(config VecEnvConfig, device consts.DeviceType) *VecEnv {
	env := &VecEnv{
		config: config,
		device: device,
	}

	// Initialize state tensors on GPU.
	env.stepIndices = tensor.Zeros(consts.KInt64, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))
	env.positions = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))
	env.entryPrices = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))
	env.equity = tensor.FromFloat32(makeSlice(config.NumEnvs, float32(config.InitialEquity)), tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))
	env.maxEquity = tensor.FromFloat32(makeSlice(config.NumEnvs, float32(config.InitialEquity)), tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))

	// Pre-allocate output buffers (state dim = feature_dim + 2 for position and equity).
	stateDim := config.FeatureDim + 2
	env.outStates = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs, stateDim), tensor.WithDevice(device))
	env.outRewards = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))
	env.outDones = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))
	env.outPositions = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))
	env.outEntryPrices = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))
	env.outEquity = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))
	env.outMaxEquity = tensor.Zeros(consts.KFloat, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))
	env.outStepIndices = tensor.Zeros(consts.KInt64, tensor.WithShapes(config.NumEnvs), tensor.WithDevice(device))

	return env
}

// LoadMarketData loads market data to GPU.
// data shape: [num_envs, time_steps, feature_dim]
// Each env can have different market sequences for diversity.
func (env *VecEnv) LoadMarketData(data *tensor.Tensor) {
	env.marketData = data.ToDevice(env.device)
	env.maxTimeSteps = data.Shapes()[1]
}

// LoadMarketDataFromSlices loads market data from Go slices.
// Creates data on GPU directly.
func (env *VecEnv) LoadMarketDataFromSlices(data []float32, timeSteps int64) {
	env.maxTimeSteps = timeSteps
	env.marketData = tensor.FromFloat32(data,
		tensor.WithShapes(env.config.NumEnvs, timeSteps, env.config.FeatureDim),
		tensor.WithDevice(env.device))
}

// Reset resets all environments.
func (env *VecEnv) Reset() *tensor.Tensor {
	// Reset state.
	torch.EnvResetDone(
		tensor.Ones(consts.KFloat, tensor.WithShapes(env.config.NumEnvs), tensor.WithDevice(env.device)).Tensor(),
		env.positions.Tensor(),
		env.entryPrices.Tensor(),
		env.equity.Tensor(),
		env.maxEquity.Tensor(),
		env.stepIndices.Tensor(),
		env.config.InitialEquity,
	)

	// Build initial state.
	return env.getCurrentState()
}

// Step takes actions and returns (next_states, rewards, dones).
// actions: [num_envs] - discrete actions (0=flat, 1=long, 2=short)
// All operations happen on GPU.
func (env *VecEnv) Step(actions *tensor.Tensor) (states, rewards, dones *tensor.Tensor) {
	// Run vectorized step on GPU.
	torch.EnvVectorizedStep(
		env.marketData.Tensor(),
		env.stepIndices.Tensor(),
		env.positions.Tensor(),
		env.entryPrices.Tensor(),
		env.equity.Tensor(),
		env.maxEquity.Tensor(),
		actions.Tensor(),
		env.config.FeeRate,
		env.config.MinEquity,
		env.config.MaxSteps,
		env.outStates.Tensor(),
		env.outRewards.Tensor(),
		env.outDones.Tensor(),
		env.outPositions.Tensor(),
		env.outEntryPrices.Tensor(),
		env.outEquity.Tensor(),
		env.outMaxEquity.Tensor(),
		env.outStepIndices.Tensor(),
	)

	// Update internal state from outputs.
	env.positions = env.outPositions
	env.entryPrices = env.outEntryPrices
	env.equity = env.outEquity
	env.maxEquity = env.outMaxEquity
	env.stepIndices = env.outStepIndices

	// Auto-reset done environments.
	torch.EnvResetDone(
		env.outDones.Tensor(),
		env.positions.Tensor(),
		env.entryPrices.Tensor(),
		env.equity.Tensor(),
		env.maxEquity.Tensor(),
		env.stepIndices.Tensor(),
		env.config.InitialEquity,
	)

	return env.outStates, env.outRewards, env.outDones
}

// getCurrentState builds the current observation state.
// Returns [num_envs, feature_dim + 2] tensor (market features + position + equity)
//
// Note: Creates intermediate tensors for narrow/squeeze. These are tracked by
// scope if active, otherwise freed explicitly.
func (env *VecEnv) getCurrentState() *tensor.Tensor {

	if env.marketData == nil {
		// No market data loaded, return zeros.
		return env.outStates
	}

	// Get current market features at step index 0 (after reset).
	// marketData shape: [num_envs, time_steps, feature_dim]
	// stepIndices after reset are 0, so we need features at index 0.

	// Use narrow to select first timestep: marketData[:, 0:1, :] then squeeze.
	// Narrow: (dim=1, start=0, length=1)
	narrowed := env.marketData.Narrow(1, 0, 1)
	currentFeatures := narrowed.Squeeze(1)
	// Result shape: [num_envs, feature_dim]

	// Build state using env_build_state kernel.
	state := tensor.New(torch.EnvBuildState(
		currentFeatures.Tensor(),
		env.positions.Tensor(),
		env.equity.Tensor(),
	))

	// Free intermediate tensors.
	narrowed.Free()
	currentFeatures.Free()

	return state
}

// GetNumEnvs returns the number of parallel environments.
func (env *VecEnv) GetNumEnvs() int64 {
	return env.config.NumEnvs
}

// GetStateDim returns the observation dimension.
func (env *VecEnv) GetStateDim() int64 {
	return env.config.FeatureDim + 2
}

// GetEquity returns current equity for all envs.
func (env *VecEnv) GetEquity() *tensor.Tensor {
	return env.equity
}

// GetPositions returns current positions for all envs.
func (env *VecEnv) GetPositions() *tensor.Tensor {
	return env.positions
}

// makeSlice creates a slice filled with a value.
func makeSlice(n int64, val float32) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = val
	}
	return s
}
