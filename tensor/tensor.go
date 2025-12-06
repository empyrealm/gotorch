package tensor

import (
	"runtime"
	"time"

	"github.com/empyrealm/gotorch/consts"
	"github.com/empyrealm/gotorch/internal/torch"
	"github.com/lwch/logging"
)

type Tensor struct {
	idx     uint64
	created time.Time
	t       torch.Tensor
}

func New(t torch.Tensor) *Tensor {
	ts := &Tensor{
		created: time.Now(),
		t:       t,
	}
	logBuildInfo(ts)
	logging.Debug("new tensor: %d", ts.idx)
	runtime.SetFinalizer(ts, freeTensor)
	return ts
}

func freeTensor(t *Tensor) error {
	if t == nil || t.t == nil {
		return nil
	}
	logging.Debug("free tensor: %d", t.idx)
	free(t)
	torch.FreeTensor(t.t)
	t.t = nil
	runtime.SetFinalizer(t, nil)
	return nil
}

func (t *Tensor) Created() time.Time {
	return t.created
}

func (t *Tensor) Tensor() torch.Tensor {
	return t.t
}

func (t *Tensor) Reshape(shape ...int64) *Tensor {
	ptr := torch.Reshape(t.t, shape)
	return New(ptr)
}

func (t *Tensor) Transpose(dim1, dim2 int64) *Tensor {
	ptr := torch.Transpose(t.t, dim1, dim2)
	return New(ptr)
}

func (t *Tensor) ElemSize() int64 {
	return torch.ElemSize(t.t)
}

func (t *Tensor) ElemCount() int64 {
	return torch.ElemCount(t.t)
}

func (t *Tensor) Dims() int64 {
	return torch.Dims(t.t)
}

func (t *Tensor) Shapes() []int64 {
	return torch.Shapes(t.t)
}

func (t *Tensor) ScalarType() consts.ScalarType {
	return torch.ScalarType(t.t)
}

func (t *Tensor) DeviceType() consts.DeviceType {
	return torch.DeviceType(t.t)
}

func (t *Tensor) SetRequiresGrad(b bool) {
	torch.SetRequiresGrad(t.t, b)
}

func (t *Tensor) ToDevice(device consts.DeviceType) *Tensor {
	ptr := torch.ToDevice(t.t, device)
	return New(ptr)
}

func (t *Tensor) ToScalarType(scalarType consts.ScalarType) *Tensor {
	ptr := torch.ToScalarType(t.t, scalarType)
	return New(ptr)
}

func (t *Tensor) Detach() *Tensor {
	ptr := torch.Detach(t.t)
	return New(ptr)
}

func (t *Tensor) Clone() *Tensor {
	ptr := torch.Clone(t.t)
	return New(ptr)
}

// Free explicitly releases the tensor's memory.
// This is useful for immediate cleanup of CUDA tensors without waiting for GC.
// Safe to call multiple times - subsequent calls are no-ops.
func (t *Tensor) Free() {
	if t == nil || t.t == nil {
		return
	}

	logging.Debug("explicit free tensor: %d", t.idx)
	free(t)
	torch.FreeTensor(t.t)
	t.t = nil
	runtime.SetFinalizer(t, nil)
}

// NanToNum replaces NaN, positive infinity, and negative infinity with specified values.
// If a value is not specified (0), uses reasonable defaults:
// - NaN -> 0.0
// - +Inf -> very large positive number
// - -Inf -> very large negative number
func (t *Tensor) NanToNum(nan, posinf, neginf float64) *Tensor {
	ptr := torch.NanToNum(t.t, nan, posinf, neginf)
	return New(ptr)
}
