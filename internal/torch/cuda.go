package torch

/*
#include "tensor.h"
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
