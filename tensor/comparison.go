// Package tensor provides comparison operations for tensors.
package tensor

import (
	"github.com/empyrealm/gotorch/internal/torch"
)

// ============================================================================
// Element-wise Comparison Operations
// ============================================================================

// Eq returns element-wise equality comparison (a == b).
func (t *Tensor) Eq(other *Tensor) *Tensor {
	return New(torch.Eq(t.t, other.t))
}

// Ne returns element-wise not-equal comparison (a != b).
func (t *Tensor) Ne(other *Tensor) *Tensor {
	return New(torch.Ne(t.t, other.t))
}

// Lt returns element-wise less-than comparison (a < b).
func (t *Tensor) Lt(other *Tensor) *Tensor {
	return New(torch.Lt(t.t, other.t))
}

// Le returns element-wise less-than-or-equal comparison (a <= b).
func (t *Tensor) Le(other *Tensor) *Tensor {
	return New(torch.Le(t.t, other.t))
}

// Gt returns element-wise greater-than comparison (a > b).
func (t *Tensor) Gt(other *Tensor) *Tensor {
	return New(torch.Gt(t.t, other.t))
}

// Ge returns element-wise greater-than-or-equal comparison (a >= b).
func (t *Tensor) Ge(other *Tensor) *Tensor {
	return New(torch.Ge(t.t, other.t))
}

// ============================================================================
// Logical Operations
// ============================================================================

// LogicalAnd returns element-wise logical AND.
func (t *Tensor) LogicalAnd(other *Tensor) *Tensor {
	return New(torch.LogicalAnd(t.t, other.t))
}

// LogicalOr returns element-wise logical OR.
func (t *Tensor) LogicalOr(other *Tensor) *Tensor {
	return New(torch.LogicalOr(t.t, other.t))
}

// LogicalNot returns element-wise logical NOT.
func (t *Tensor) LogicalNot() *Tensor {
	return New(torch.LogicalNot(t.t))
}

// ============================================================================
// Concatenation - method form
// ============================================================================

// Cat concatenates tensors along dimension 1 (column-wise).
// This is a convenience method for horizontal concatenation.
func (t *Tensor) Cat(others ...*Tensor) *Tensor {
	tensors := make([]torch.Tensor, len(others)+1)
	tensors[0] = t.t
	for i, other := range others {
		tensors[i+1] = other.t
	}
	return New(torch.Cat(tensors, 1))
}

// CatDim concatenates tensors along specified dimension.
func (t *Tensor) CatDim(dim int, others ...*Tensor) *Tensor {
	tensors := make([]torch.Tensor, len(others)+1)
	tensors[0] = t.t
	for i, other := range others {
		tensors[i+1] = other.t
	}
	return New(torch.Cat(tensors, dim))
}

// ============================================================================
// Element-wise Min/Max
// ============================================================================

// Maximum returns element-wise maximum with another tensor.
func (t *Tensor) Maximum(other *Tensor) *Tensor {
	return New(torch.Maximum(t.t, other.t))
}

// Minimum returns element-wise minimum with another tensor.
func (t *Tensor) Minimum(other *Tensor) *Tensor {
	return New(torch.Minimum(t.t, other.t))
}
