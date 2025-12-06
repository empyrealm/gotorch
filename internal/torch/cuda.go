package torch

// #include "tensor.h"
import "C"

// IsCUDA returns true if CUDA is available.
func IsCUDA() bool {
	return bool(C.is_cuda_available())
}

// CUDASynchronize synchronizes all CUDA streams.
func CUDASynchronize() {
	C.cuda_synchronize()
}

// CUDAEmptyCache releases all unused cached memory.
func CUDAEmptyCache() {
	C.cuda_empty_cache()
}

// GetCUDAMemoryAllocated returns current GPU memory usage.
func GetCUDAMemoryAllocated() uint64 {
	return uint64(C.cuda_memory_allocated())
}

// GetCUDAMemoryTotal returns total GPU memory.
func GetCUDAMemoryTotal() uint64 {
	return uint64(C.cuda_memory_total())
}

// GetCUDAMemoryFree returns free GPU memory.
func GetCUDAMemoryFree() uint64 {
	total := GetCUDAMemoryTotal()
	allocated := GetCUDAMemoryAllocated()

	if total > allocated {
		return total - allocated
	}

	return 0
}

// GetCUDADeviceName returns the CUDA device name.
func GetCUDADeviceName() string {
	return C.GoString(C.cuda_device_name())
}

// GetCUDASMVersion returns the CUDA SM version.
func GetCUDASMVersion() string {
	return C.GoString(C.cuda_sm_version())
}
