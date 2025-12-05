package torch

// #include "tensor.h"
import "C"

import "github.com/empyrealm/gotorch/consts"

// Multinomial samples from multinomial distribution on GPU.
func Multinomial(probs Tensor, numSamples int64, replacement bool) Tensor {
	var err *C.char
	ret := C.tensor_multinomial(&err, C.tensor(probs), C.int64_t(numSamples), C.bool(replacement))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// CategoricalSample samples from categorical distribution (softmax + multinomial).
func CategoricalSample(logits Tensor) Tensor {
	var err *C.char
	ret := C.tensor_categorical_sample(&err, C.tensor(logits))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// NormalSample samples from normal distribution on GPU.
func NormalSample(mean, std Tensor) Tensor {
	var err *C.char
	ret := C.tensor_normal_sample(&err, C.tensor(mean), C.tensor(std))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// Argmax returns indices of maximum values.
func Argmax(t Tensor, dim int64, keepdim bool) Tensor {
	var err *C.char
	ret := C.tensor_argmax(&err, C.tensor(t), C.int64_t(dim), C.bool(keepdim))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// Rand creates a random uniform tensor on device.
func Rand(shape []int64, device consts.DeviceType) Tensor {
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ret := C.tensor_rand(&err, shapes, size, C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// Randn creates a random normal tensor on device.
func Randn(shape []int64, device consts.DeviceType) Tensor {
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ret := C.tensor_randn(&err, shapes, size, C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// ClampMinMax clamps tensor values between min and max.
func ClampMinMax(t Tensor, minVal, maxVal float64) Tensor {
	var err *C.char
	ret := C.tensor_clamp_minmax(&err, C.tensor(t), C.double(minVal), C.double(maxVal))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// Where selects elements from x or y based on condition.
func Where(condition, x, y Tensor) Tensor {
	var err *C.char
	ret := C.tensor_where(&err, C.tensor(condition), C.tensor(x), C.tensor(y))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}
