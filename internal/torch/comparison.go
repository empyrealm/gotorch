package torch

// #include "tensor.h"
import "C"

// ============================================================================
// Element-wise Comparison Operations
// ============================================================================

// Eq returns element-wise equality comparison.
func Eq(a, b Tensor) Tensor {
	var err *C.char
	ret := C.tensor_eq(&err, C.tensor(a), C.tensor(b))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// Ne returns element-wise not-equal comparison.
func Ne(a, b Tensor) Tensor {
	var err *C.char
	ret := C.tensor_ne(&err, C.tensor(a), C.tensor(b))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// Lt returns element-wise less-than comparison.
func Lt(a, b Tensor) Tensor {
	var err *C.char
	ret := C.tensor_lt(&err, C.tensor(a), C.tensor(b))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// Le returns element-wise less-than-or-equal comparison.
func Le(a, b Tensor) Tensor {
	var err *C.char
	ret := C.tensor_le(&err, C.tensor(a), C.tensor(b))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// Gt returns element-wise greater-than comparison.
func Gt(a, b Tensor) Tensor {
	var err *C.char
	ret := C.tensor_gt(&err, C.tensor(a), C.tensor(b))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// Ge returns element-wise greater-than-or-equal comparison.
func Ge(a, b Tensor) Tensor {
	var err *C.char
	ret := C.tensor_ge(&err, C.tensor(a), C.tensor(b))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// ============================================================================
// Logical Operations
// ============================================================================

// LogicalAnd returns element-wise logical AND.
func LogicalAnd(a, b Tensor) Tensor {
	var err *C.char
	ret := C.tensor_logical_and(&err, C.tensor(a), C.tensor(b))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// LogicalOr returns element-wise logical OR.
func LogicalOr(a, b Tensor) Tensor {
	var err *C.char
	ret := C.tensor_logical_or(&err, C.tensor(a), C.tensor(b))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// LogicalNot returns element-wise logical NOT.
func LogicalNot(t Tensor) Tensor {
	var err *C.char
	ret := C.tensor_logical_not(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// ============================================================================
// Min/Max between tensors
// ============================================================================

// Maximum returns element-wise maximum of two tensors.
func Maximum(a, b Tensor) Tensor {
	var err *C.char
	ret := C.tensor_maximum(&err, C.tensor(a), C.tensor(b))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// Minimum returns element-wise minimum of two tensors.
func Minimum(a, b Tensor) Tensor {
	var err *C.char
	ret := C.tensor_minimum(&err, C.tensor(a), C.tensor(b))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}
