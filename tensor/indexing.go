package tensor

import (
	"github.com/empyrealm/gotorch/consts"
	"github.com/empyrealm/gotorch/internal/torch"
)

// ============================================================================
// Indexing Operations (critical for GPU replay buffers)
// All operations use New() for proper TensorScope tracking.
// ============================================================================

// Index reads from tensor at given indices: t[i, j, ...]
func (t *Tensor) Index(indices []int64) *Tensor {
	return New(torch.Index(t.t, indices))
}

// IndexPut writes to tensor at given indices: t[i, j, ...] = value
// Modifies tensor in-place.
func (t *Tensor) IndexPut(indices []int64, value *Tensor) {
	torch.IndexPut(t.t, indices, value.t)
}

// IndexPutTensor writes using tensor indices: t[indices_tensor] = value
// Useful for scatter operations in prioritized replay.
func (t *Tensor) IndexPutTensor(indices *Tensor, value *Tensor) {
	torch.IndexPutTensor(t.t, indices.t, value.t)
}

// IndexSelect selects elements along dimension using index tensor.
// Stays on GPU - critical for batch sampling.
func (t *Tensor) IndexSelect(dim int64, indices *Tensor) *Tensor {
	return New(torch.IndexSelect(t.t, dim, indices.t))
}

// Narrow slices tensor along dimension.
func (t *Tensor) Narrow(dim, start, length int64) *Tensor {
	return New(torch.NArrow(t.t, dim, start, length))
}

// ReshapeSlice changes tensor shape (slice argument version).
func (t *Tensor) ReshapeSlice(shape []int64) *Tensor {
	return New(torch.Reshape(t.t, shape))
}

// ============================================================================
// Global Reductions (over all elements)
// ============================================================================

// MeanAll computes mean over all elements (scalar result).
func (t *Tensor) MeanAll() *Tensor {
	return New(torch.MeanAll(t.t))
}

// SumAll computes sum over all elements (scalar result).
func (t *Tensor) SumAll() *Tensor {
	return New(torch.SumAll(t.t))
}

// MaxAll returns maximum over all elements (scalar result).
func (t *Tensor) MaxAll() *Tensor {
	return New(torch.MaxAll(t.t))
}

// MinAll returns minimum over all elements (scalar result).
func (t *Tensor) MinAll() *Tensor {
	return New(torch.MinAll(t.t))
}

// StdAll computes standard deviation over all elements.
func (t *Tensor) StdAll(unbiased bool) *Tensor {
	return New(torch.StdAll(t.t, unbiased))
}

// PowTensor raises t to power of exp tensor element-wise.
func (t *Tensor) PowTensor(exp *Tensor) *Tensor {
	return New(torch.PowTensor(t.t, exp.t))
}

// ============================================================================
// Tensor Creation
// ============================================================================

// Ones creates tensor filled with ones on specified device.
func Ones(dtype consts.ScalarType, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	return New(torch.Ones(args.shapes, dtype, args.device))
}
