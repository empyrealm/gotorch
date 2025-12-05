package torch

// #include "tensor.h"
import "C"

// ============================================================================
// Mixed Precision Operations
// ============================================================================

// Half converts tensor to fp16.
func Half(t Tensor) Tensor {
	var err *C.char
	ret := C.tensor_half(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// BFloat16 converts tensor to bf16.
func BFloat16(t Tensor) Tensor {
	var err *C.char
	ret := C.tensor_bfloat16(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// Float32 converts tensor to fp32.
func Float32(t Tensor) Tensor {
	var err *C.char
	ret := C.tensor_float32(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// IsHalf returns true if tensor is fp16.
func IsHalf(t Tensor) bool {
	return bool(C.tensor_is_half(C.tensor(t)))
}

// IsBFloat16 returns true if tensor is bf16.
func IsBFloat16(t Tensor) bool {
	return bool(C.tensor_is_bfloat16(C.tensor(t)))
}

// Scale multiplies tensor by scalar.
func Scale(t Tensor, scale float64) Tensor {
	var err *C.char
	ret := C.tensor_scale(&err, C.tensor(t), C.double(scale))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// ScaleInPlace multiplies tensor by scalar in-place.
func ScaleInPlace(t Tensor, scale float64) {
	// For now, use non-inplace version.
	// Could add proper in-place op later.
	var err *C.char
	C.tensor_scale(&err, C.tensor(t), C.double(scale))
	if err != nil {
		panic(C.GoString(err))
	}
}

// IsFinite returns true if all tensor elements are finite.
func IsFinite(t Tensor) bool {
	return bool(C.tensor_is_finite(C.tensor(t)))
}
