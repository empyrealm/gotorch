package torch

// #include "tensor.h"
import "C"

import "github.com/empyrealm/gotorch/consts"

// ============================================================================
// Indexing Operations
// ============================================================================

// Index reads from tensor at given indices: t[i, j, ...]
func Index(t Tensor, indices []int64) Tensor {
	idx, size := cInts[int64, C.int64_t](indices)
	var err *C.char
	ret := C.tensor_index(&err, C.tensor(t), idx, size)
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// IndexPut writes to tensor at given indices: t[i, j, ...] = value
func IndexPut(t Tensor, indices []int64, value Tensor) {
	idx, size := cInts[int64, C.int64_t](indices)
	var err *C.char
	C.tensor_index_put(&err, C.tensor(t), idx, size, C.tensor(value))
	if err != nil {
		panic(C.GoString(err))
	}
}

// IndexPutTensor writes using tensor indices: t[indices_tensor] = value
func IndexPutTensor(t Tensor, indices Tensor, value Tensor) {
	var err *C.char
	C.tensor_index_put_tensor(&err, C.tensor(t), C.tensor(indices), C.tensor(value))
	if err != nil {
		panic(C.GoString(err))
	}
}

// IndexSelect selects along dimension using index tensor.
func IndexSelect(t Tensor, dim int64, indices Tensor) Tensor {
	var err *C.char
	ret := C.tensor_index_select(&err, C.tensor(t), C.int64_t(dim), C.tensor(indices))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// Narrow is defined in shapes.go as NArrow - use that instead.

// ============================================================================
// Global Reduction Operations
// ============================================================================

// MeanAll computes mean over all elements.
func MeanAll(t Tensor) Tensor {
	var err *C.char
	ret := C.tensor_mean_all(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// SumAll computes sum over all elements.
func SumAll(t Tensor) Tensor {
	var err *C.char
	ret := C.tensor_sum_all(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// MaxAll returns maximum over all elements.
func MaxAll(t Tensor) Tensor {
	var err *C.char
	ret := C.tensor_max_all(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// MinAll returns minimum over all elements.
func MinAll(t Tensor) Tensor {
	var err *C.char
	ret := C.tensor_min_all(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// StdAll computes standard deviation over all elements.
func StdAll(t Tensor, unbiased bool) Tensor {
	var err *C.char
	ret := C.tensor_std_all(&err, C.tensor(t), C.bool(unbiased))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// PowTensor computes t^exp element-wise.
func PowTensor(t, exp Tensor) Tensor {
	var err *C.char
	ret := C.tensor_pow_tensor(&err, C.tensor(t), C.tensor(exp))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// Ones creates tensor filled with ones.
func Ones(shape []int64, dtype consts.ScalarType, device consts.DeviceType) Tensor {
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ret := C.tensor_ones(&err, shapes, size, C.int8_t(dtype), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}
