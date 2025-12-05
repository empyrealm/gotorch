// Package tensor provides mixed precision (AMP) support.
package tensor

import (
	"github.com/empyrealm/gotorch/internal/torch"
)

// ============================================================================
// Mixed Precision Conversions
// ============================================================================

// Half converts tensor to fp16 (half precision).
// 2x memory savings, faster on Tensor Cores (RTX GPUs).
func (t *Tensor) Half() *Tensor {
	return &Tensor{t: torch.Half(t.t)}
}

// BFloat16 converts tensor to bf16.
// Better numerical stability than fp16, same memory savings.
func (t *Tensor) BFloat16() *Tensor {
	return &Tensor{t: torch.BFloat16(t.t)}
}

// Float32 converts tensor back to fp32 (full precision).
func (t *Tensor) Float32() *Tensor {
	return &Tensor{t: torch.Float32(t.t)}
}

// IsHalf returns true if tensor is fp16.
func (t *Tensor) IsHalf() bool {
	return torch.IsHalf(t.t)
}

// IsBFloat16 returns true if tensor is bf16.
func (t *Tensor) IsBFloat16() bool {
	return torch.IsBFloat16(t.t)
}

// ============================================================================
// Gradient Scaler for AMP Training
// ============================================================================

// GradScaler implements gradient scaling for mixed precision training.
// Prevents fp16 gradients from underflowing to zero.
type GradScaler struct {
	scale            float64
	growthFactor     float64
	backoffFactor    float64
	growthInterval   int
	stepsSinceGrowth int
	foundInf         bool
}

// NewGradScaler creates a gradient scaler for AMP training.
func NewGradScaler() *GradScaler {
	return &GradScaler{
		scale:            65536.0, // 2^16 - good starting point for fp16
		growthFactor:     2.0,
		backoffFactor:    0.5,
		growthInterval:   2000, // Steps between scale increases
		stepsSinceGrowth: 0,
		foundInf:         false,
	}
}

// Scale multiplies the loss by the current scale factor.
// Call this before backward().
func (s *GradScaler) Scale(loss *Tensor) *Tensor {
	return loss.MulScalar(s.scale)
}

// Unscale divides gradients by scale factor.
// Call this after backward(), before optimizer step.
func (s *GradScaler) Unscale(grads []*Tensor) {
	invScale := 1.0 / s.scale
	for _, g := range grads {
		if g != nil {
			torch.ScaleInPlace(g.t, invScale)
		}
	}
}

// Step checks for inf/nan gradients and updates scale.
// Returns true if gradients are valid and optimizer should step.
func (s *GradScaler) Step(grads []*Tensor) bool {
	// Check for inf/nan in any gradient.
	s.foundInf = false
	for _, g := range grads {
		if g != nil && !torch.IsFinite(g.t) {
			s.foundInf = true
			break
		}
	}

	if s.foundInf {
		// Reduce scale on inf/nan.
		s.scale *= s.backoffFactor
		s.stepsSinceGrowth = 0
		return false // Skip optimizer step.
	}

	// No inf/nan - consider growing scale.
	s.stepsSinceGrowth++
	if s.stepsSinceGrowth >= s.growthInterval {
		s.scale *= s.growthFactor
		s.stepsSinceGrowth = 0
	}

	return true // Safe to step optimizer.
}

// GetScale returns current scale factor.
func (s *GradScaler) GetScale() float64 {
	return s.scale
}

// ============================================================================
// AMP Context Manager
// ============================================================================

// AMPConfig configures automatic mixed precision.
type AMPConfig struct {
	Enabled   bool    // Enable AMP
	UseBF16   bool    // Use bf16 instead of fp16
	InitScale float64 // Initial gradient scale
}

// DefaultAMPConfig returns sensible defaults for AMP.
func DefaultAMPConfig() AMPConfig {
	return AMPConfig{
		Enabled:   true,
		UseBF16:   false, // fp16 is faster on most GPUs
		InitScale: 65536.0,
	}
}

// MulScalar multiplies tensor by scalar (helper for scaling).
func (t *Tensor) MulScalar(s float64) *Tensor {
	return &Tensor{t: torch.Scale(t.t, s)}
}
