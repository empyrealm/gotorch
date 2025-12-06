package torch

/*
#include "tensor.h"
#include <stdlib.h>
*/
import "C"


// CUDAEmptyCache clears the CUDA memory cache.
func CUDAEmptyCache() {
	C.cuda_empty_cache()
}


// CUDASynchronize waits for all CUDA operations to complete.
func CUDASynchronize() {
	C.cuda_synchronize()
}


// CUDAMemoryAllocated returns the current CUDA memory allocated in bytes.
func CUDAMemoryAllocated() int64 {
	return int64(C.cuda_memory_allocated())
}


// CUDAMemoryReserved returns the current CUDA memory reserved in bytes.
func CUDAMemoryReserved() int64 {
	return int64(C.cuda_memory_reserved())
}


// CUDAMemoryTotal returns the total CUDA memory in bytes.
func CUDAMemoryTotal() int64 {
	return int64(C.cuda_memory_total())
}


// CUDAMemoryFree returns the free CUDA memory in bytes.
func CUDAMemoryFree() int64 {
	return int64(C.cuda_memory_free())
}


// CUDADeviceCount returns the number of CUDA devices.
func CUDADeviceCount() int {
	return int(C.cuda_device_count())
}


// CUDADeviceName returns the name of a CUDA device.
func CUDADeviceName(deviceID int) string {
	return C.GoString(C.cuda_device_name(C.int(deviceID)))
}


// CUDAComputeCapability returns the compute capability (e.g., 89 for sm_89).
func CUDAComputeCapability(deviceID int) int {
	return int(C.cuda_compute_capability(C.int(deviceID)))
}
