package tensor

import "github.com/empyrealm/gotorch/internal/torch"


// EmptyCache clears the CUDA memory cache.
// Call this periodically during long training runs to prevent OOM.
func EmptyCache() {
	torch.CUDAEmptyCache()
}


// Synchronize waits for all CUDA operations to complete.
func Synchronize() {
	torch.CUDASynchronize()
}


// MemoryAllocated returns the current CUDA memory allocated in bytes.
func MemoryAllocated() int64 {
	return torch.CUDAMemoryAllocated()
}


// MemoryReserved returns the current CUDA memory reserved in bytes.
func MemoryReserved() int64 {
	return torch.CUDAMemoryReserved()
}
