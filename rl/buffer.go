// Package rl provides reinforcement learning utilities for gotorch.
package rl

import (
	"github.com/empyrealm/gotorch/consts"
	"github.com/empyrealm/gotorch/tensor"
)

// ============================================================================
// CUDA Replay Buffer
// Keeps all experience data on GPU to avoid transfer bottlenecks.
// ============================================================================

// RolloutBuffer stores PPO rollout data entirely on GPU.
type RolloutBuffer struct {
	device consts.DeviceType

	// Buffer dimensions.
	numEnvs   int64
	numSteps  int64
	stateDim  int64
	actionDim int64

	// GPU tensors (pre-allocated).
	states     *tensor.Tensor // [numSteps, numEnvs, stateDim]
	actions    *tensor.Tensor // [numSteps, numEnvs, actionDim]
	rewards    *tensor.Tensor // [numSteps, numEnvs]
	dones      *tensor.Tensor // [numSteps, numEnvs]
	values     *tensor.Tensor // [numSteps, numEnvs]
	logProbs   *tensor.Tensor // [numSteps, numEnvs]
	advantages *tensor.Tensor // [numSteps, numEnvs]
	returns    *tensor.Tensor // [numSteps, numEnvs]

	// Current position.
	ptr int64
}

// NewRolloutBuffer creates a new GPU rollout buffer.
func NewRolloutBuffer(numEnvs, numSteps, stateDim, actionDim int64, device consts.DeviceType) *RolloutBuffer {
	buf := &RolloutBuffer{
		device:    device,
		numEnvs:   numEnvs,
		numSteps:  numSteps,
		stateDim:  stateDim,
		actionDim: actionDim,
		ptr:       0,
	}

	// Pre-allocate GPU tensors.
	buf.states = tensor.Zeros(consts.KFloat, tensor.WithShapes(numSteps, numEnvs, stateDim), tensor.WithDevice(device))
	buf.actions = tensor.Zeros(consts.KFloat, tensor.WithShapes(numSteps, numEnvs, actionDim), tensor.WithDevice(device))
	buf.rewards = tensor.Zeros(consts.KFloat, tensor.WithShapes(numSteps, numEnvs), tensor.WithDevice(device))
	buf.dones = tensor.Zeros(consts.KFloat, tensor.WithShapes(numSteps, numEnvs), tensor.WithDevice(device))
	buf.values = tensor.Zeros(consts.KFloat, tensor.WithShapes(numSteps, numEnvs), tensor.WithDevice(device))
	buf.logProbs = tensor.Zeros(consts.KFloat, tensor.WithShapes(numSteps, numEnvs), tensor.WithDevice(device))
	buf.advantages = tensor.Zeros(consts.KFloat, tensor.WithShapes(numSteps, numEnvs), tensor.WithDevice(device))
	buf.returns = tensor.Zeros(consts.KFloat, tensor.WithShapes(numSteps, numEnvs), tensor.WithDevice(device))

	return buf
}

// Reset clears the buffer for a new rollout.
func (b *RolloutBuffer) Reset() {
	b.ptr = 0
}

// Add adds a transition to the buffer (tensors must already be on GPU).
func (b *RolloutBuffer) Add(state, action, reward, done, value, logProb *tensor.Tensor) {
	if b.ptr >= b.numSteps {
		return // Buffer full.
	}

	// Copy into buffer at current position.
	// Using index assignment (all on GPU).
	b.states.IndexPut([]int64{b.ptr}, state)
	b.actions.IndexPut([]int64{b.ptr}, action)
	b.rewards.IndexPut([]int64{b.ptr}, reward)
	b.dones.IndexPut([]int64{b.ptr}, done)
	b.values.IndexPut([]int64{b.ptr}, value)
	b.logProbs.IndexPut([]int64{b.ptr}, logProb)

	b.ptr++
}

// ComputeReturnsAndAdvantages computes GAE on GPU with proper memory management.
//
// Uses TensorScope to track and free intermediate tensors, preventing memory
// leaks during the backward GAE pass.
func (b *RolloutBuffer) ComputeReturnsAndAdvantages(lastValue *tensor.Tensor, gamma, gaeLambda float64) {

	// Create scalar tensors on GPU (reused across all iterations).
	gammaT := tensor.FromFloat32([]float32{float32(gamma)}, tensor.WithShapes(1), tensor.WithDevice(b.device))
	lambdaT := tensor.FromFloat32([]float32{float32(gaeLambda)}, tensor.WithShapes(1), tensor.WithDevice(b.device))
	oneT := tensor.FromFloat32([]float32{1.0}, tensor.WithShapes(1), tensor.WithDevice(b.device))
	gammaLambdaT := gammaT.Mul(lambdaT) // Pre-compute gamma * lambda.

	// Initialize last advantage.
	lastAdv := tensor.Zeros(consts.KFloat, tensor.WithShapes(b.numEnvs), tensor.WithDevice(b.device))
	nextValue := lastValue

	// Backward pass for GAE (on GPU).
	// Each iteration creates intermediate tensors that must be freed.
	for t := b.numSteps - 1; t >= 0; t-- {

		// Scope for this iteration - frees intermediates automatically.
		scope := tensor.NewScope()

		// Get current step data (views into pre-allocated buffer, but Index creates new tensor).
		reward := b.rewards.Index([]int64{t})
		done := b.dones.Index([]int64{t})
		value := b.values.Index([]int64{t})

		// delta = reward + gamma * next_value * (1 - done) - value
		notDone := oneT.Sub(done)
		gammaNV := gammaT.Mul(nextValue)
		gammaNVND := gammaNV.Mul(notDone)
		rewardPlusDiscount := reward.Add(gammaNVND)
		delta := rewardPlusDiscount.Sub(value)

		// advantage = delta + gamma * lambda * (1 - done) * last_advantage
		glND := gammaLambdaT.Mul(notDone)
		glNDLA := glND.Mul(lastAdv)
		newAdv := delta.Add(glNDLA)

		// Compute return for storage.
		newReturn := newAdv.Add(value)

		// Store in pre-allocated buffers (in-place, no new allocation).
		b.advantages.IndexPut([]int64{t}, newAdv)
		b.returns.IndexPut([]int64{t}, newReturn)

		// Keep tensors needed for next iteration.
		scope.Keep(newAdv)
		scope.Keep(value)

		// Close scope - frees all intermediates except kept ones.
		scope.Close()

		// Update for next iteration.
		lastAdv = newAdv
		nextValue = value
	}

	// Free the scalar tensors used in the loop.
	gammaT.Free()
	lambdaT.Free()
	oneT.Free()
	gammaLambdaT.Free()
	lastAdv.Free()

	// Normalize advantages on GPU (with scoped intermediates).
	normScope := tensor.NewScope()

	advMean := b.advantages.MeanAll()
	advStd := b.advantages.StdAll(false)
	eps := tensor.FromFloat32([]float32{1e-8}, tensor.WithShapes(1), tensor.WithDevice(b.device))
	stdPlusEps := advStd.Add(eps)
	advCentered := b.advantages.Sub(advMean)
	normalizedAdv := advCentered.Div(stdPlusEps)

	// Keep the normalized result.
	normScope.Keep(normalizedAdv)
	normScope.Close()

	// Replace advantages with normalized version.
	b.advantages.Free()
	b.advantages = normalizedAdv
}

// GetBatch returns a minibatch from the flattened buffer.
// All tensors stay on GPU. Caller is responsible for freeing returned tensors.
//
// Note: The reshape operations create views (no copy), but IndexSelect creates
// new tensors. The returned tensors should be freed after use or tracked in a scope.
func (b *RolloutBuffer) GetBatch(indices *tensor.Tensor) (states, actions, oldLogProbs, advantages, returns *tensor.Tensor) {

	// Flatten buffer (these are views, not copies).
	flatStates := b.states.ReshapeSlice([]int64{-1, b.stateDim})
	flatActions := b.actions.ReshapeSlice([]int64{-1, b.actionDim})
	flatLogProbs := b.logProbs.ReshapeSlice([]int64{-1})
	flatAdvantages := b.advantages.ReshapeSlice([]int64{-1})
	flatReturns := b.returns.ReshapeSlice([]int64{-1})

	// Index select (on GPU) - creates new tensors.
	states = flatStates.IndexSelect(0, indices)
	actions = flatActions.IndexSelect(0, indices)
	oldLogProbs = flatLogProbs.IndexSelect(0, indices)
	advantages = flatAdvantages.IndexSelect(0, indices)
	returns = flatReturns.IndexSelect(0, indices)

	// Free the intermediate reshape views.
	flatStates.Free()
	flatActions.Free()
	flatLogProbs.Free()
	flatAdvantages.Free()
	flatReturns.Free()

	return
}

// Size returns total number of transitions.
func (b *RolloutBuffer) Size() int64 {
	return b.ptr * b.numEnvs
}

// IsFull returns true if buffer is full.
func (b *RolloutBuffer) IsFull() bool {
	return b.ptr >= b.numSteps
}

// ============================================================================
// Prioritized Replay Buffer (for DQN/SAC)
// ============================================================================

// PrioritizedBuffer implements prioritized experience replay on GPU.
type PrioritizedBuffer struct {
	device consts.DeviceType

	capacity  int64
	stateDim  int64
	actionDim int64

	// GPU storage.
	states     *tensor.Tensor // [capacity, stateDim]
	actions    *tensor.Tensor // [capacity, actionDim]
	rewards    *tensor.Tensor // [capacity]
	nextStates *tensor.Tensor // [capacity, stateDim]
	dones      *tensor.Tensor // [capacity]
	priorities *tensor.Tensor // [capacity]

	ptr  int64
	size int64

	alpha float64 // Priority exponent.
	beta  float64 // Importance sampling exponent.
}

// NewPrioritizedBuffer creates a prioritized replay buffer on GPU.
func NewPrioritizedBuffer(capacity, stateDim, actionDim int64, alpha, beta float64, device consts.DeviceType) *PrioritizedBuffer {
	return &PrioritizedBuffer{
		device:     device,
		capacity:   capacity,
		stateDim:   stateDim,
		actionDim:  actionDim,
		states:     tensor.Zeros(consts.KFloat, tensor.WithShapes(capacity, stateDim), tensor.WithDevice(device)),
		actions:    tensor.Zeros(consts.KFloat, tensor.WithShapes(capacity, actionDim), tensor.WithDevice(device)),
		rewards:    tensor.Zeros(consts.KFloat, tensor.WithShapes(capacity), tensor.WithDevice(device)),
		nextStates: tensor.Zeros(consts.KFloat, tensor.WithShapes(capacity, stateDim), tensor.WithDevice(device)),
		dones:      tensor.Zeros(consts.KFloat, tensor.WithShapes(capacity), tensor.WithDevice(device)),
		priorities: tensor.Ones(consts.KFloat, tensor.WithShapes(capacity), tensor.WithDevice(device)),
		ptr:        0,
		size:       0,
		alpha:      alpha,
		beta:       beta,
	}
}

// Add adds a transition with max priority.
func (b *PrioritizedBuffer) Add(state, action, reward, nextState, done *tensor.Tensor) {
	// Get max priority.
	maxPriority := b.priorities.MaxAll()

	// Store at current position.
	b.states.IndexPut([]int64{b.ptr}, state)
	b.actions.IndexPut([]int64{b.ptr}, action)
	b.rewards.IndexPut([]int64{b.ptr}, reward)
	b.nextStates.IndexPut([]int64{b.ptr}, nextState)
	b.dones.IndexPut([]int64{b.ptr}, done)
	b.priorities.IndexPut([]int64{b.ptr}, maxPriority)

	b.ptr = (b.ptr + 1) % b.capacity
	if b.size < b.capacity {
		b.size++
	}
}

// Sample returns a prioritized batch (all on GPU).
func (b *PrioritizedBuffer) Sample(batchSize int64) (states, actions, rewards, nextStates, dones, weights, indices *tensor.Tensor) {
	// Get priorities for valid entries.
	validPriorities := b.priorities.Narrow(0, 0, b.size)

	// Compute sampling probabilities: P(i) = p_i^alpha / sum(p^alpha)
	probs := validPriorities.Pow(b.alpha)
	probs = probs.Div(probs.SumAll())

	// Sample indices on GPU.
	indices = probs.Multinomial(batchSize, false)

	// Compute importance sampling weights.
	// w_i = (N * P(i))^(-beta)
	nT := tensor.FromFloat32([]float32{float32(b.size)}, tensor.WithShapes(1), tensor.WithDevice(b.device))
	selectedProbs := probs.IndexSelect(0, indices)
	weights = nT.Mul(selectedProbs).Pow(-b.beta)
	weights = weights.Div(weights.MaxAll()) // Normalize by max weight.

	// Get samples.
	states = b.states.IndexSelect(0, indices)
	actions = b.actions.IndexSelect(0, indices)
	rewards = b.rewards.IndexSelect(0, indices)
	nextStates = b.nextStates.IndexSelect(0, indices)
	dones = b.dones.IndexSelect(0, indices)

	return
}

// UpdatePriorities updates priorities for sampled transitions.
func (b *PrioritizedBuffer) UpdatePriorities(indices *tensor.Tensor, tdErrors *tensor.Tensor) {
	// New priorities = |TD error| + epsilon.
	eps := tensor.FromFloat32([]float32{1e-6}, tensor.WithShapes(1), tensor.WithDevice(b.device))
	newPriorities := tdErrors.Abs().Add(eps)

	// Update at indices (scatter on GPU).
	b.priorities.IndexPutTensor(indices, newPriorities)
}

// Size returns current buffer size.
func (b *PrioritizedBuffer) Size() int64 {
	return b.size
}
