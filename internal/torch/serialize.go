package torch

// #include "tensor.h"
// #include <stdlib.h>
import "C"
import "unsafe"

// SaveTensors saves a list of tensors to a .pt file (PyTorch format).
func SaveTensors(tensors []Tensor, path string) {
	if len(tensors) == 0 {
		return
	}

	cTensors := make([]C.tensor, len(tensors))
	for i, t := range tensors {
		cTensors[i] = C.tensor(t)
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var err *C.char
	C.tensors_save(&err, (*C.tensor)(unsafe.Pointer(&cTensors[0])), C.size_t(len(cTensors)), cPath)
	if err != nil {
		panic(C.GoString(err))
	}
}

// LoadTensors loads tensors from a .pt file (PyTorch format).
func LoadTensors(path string) []Tensor {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var outTensors *C.tensor
	var err *C.char

	count := C.tensors_load(&err, cPath, &outTensors)
	if err != nil {
		panic(C.GoString(err))
	}

	if count == 0 {
		return nil
	}

	// Convert C array to Go slice.
	tensors := make([]Tensor, count)
	cSlice := unsafe.Slice(outTensors, count)
	for i := C.size_t(0); i < count; i++ {
		tensors[i] = Tensor(cSlice[i])
	}

	// Free the C array (but not the tensors themselves).
	C.free(unsafe.Pointer(outTensors))

	return tensors
}
