package tensor

import (
	"github.com/empyrealm/gotorch/consts"
	"github.com/empyrealm/gotorch/internal/torch"
)

// ============================================================================
// GPU-side Sampling Operations
// These operations stay on GPU to avoid CPU transfer bottlenecks in RL.
// ============================================================================

// Multinomial samples indices from multinomial distribution.
// Stays entirely on GPU - critical for RL action selection.
//
// Args:
//
//	numSamples: Number of samples to draw per row.
//	replacement: Whether to sample with replacement.
//
// Returns:
//
//	Tensor of sampled indices (int64).
func (t *Tensor) Multinomial(numSamples int64, replacement bool) *Tensor {
	return &Tensor{t: torch.Multinomial(t.t, numSamples, replacement)}
}

// CategoricalSample samples from categorical distribution.
// Applies softmax to logits then samples - all on GPU.
//
// Returns:
//
//	Tensor of sampled indices (int64), one per row.
func (t *Tensor) CategoricalSample() *Tensor {
	return &Tensor{t: torch.CategoricalSample(t.t)}
}

// NormalSample samples from normal distribution with given mean and std.
// Both inputs and output stay on GPU.
func NormalSample(mean, std *Tensor) *Tensor {
	return &Tensor{t: torch.NormalSample(mean.t, std.t)}
}

// Argmax returns indices of maximum values along dimension.
// Stays on GPU - useful for greedy action selection.
func (t *Tensor) Argmax(dim int64, keepdim bool) *Tensor {
	return &Tensor{t: torch.Argmax(t.t, dim, keepdim)}
}

// Rand creates a tensor with uniform random values in [0, 1).
// Created directly on specified device (GPU).
func Rand(shape []int64, device consts.DeviceType) *Tensor {
	return &Tensor{t: torch.Rand(shape, device)}
}

// Randn creates a tensor with standard normal random values.
// Created directly on specified device (GPU).
func Randn(shape []int64, device consts.DeviceType) *Tensor {
	return &Tensor{t: torch.Randn(shape, device)}
}

// RandLike creates a tensor with uniform random values matching t's shape/device.
func RandLike(t *Tensor) *Tensor {
	shape := t.Shapes()
	device := t.DeviceType()
	return Rand(shape, device)
}

// RandnLike creates a tensor with normal random values matching t's shape/device.
func RandnLike(t *Tensor) *Tensor {
	shape := t.Shapes()
	device := t.DeviceType()
	return Randn(shape, device)
}

// ClampMinMax clamps tensor values to [min, max] range.
// Useful for bounding actions in continuous action spaces.
func (t *Tensor) ClampMinMax(minVal, maxVal float64) *Tensor {
	return &Tensor{t: torch.ClampMinMax(t.t, minVal, maxVal)}
}

// Where selects elements: condition ? x : y
// All operations on GPU.
func Where(condition, x, y *Tensor) *Tensor {
	return &Tensor{t: torch.Where(condition.t, x.t, y.t)}
}
