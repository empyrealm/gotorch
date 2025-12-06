// Package tensor provides CUDA utility functions.
package tensor

import (
	"github.com/empyrealm/gotorch/internal/torch"
)

// CUDAInfo holds information about CUDA device.
type CUDAInfo struct {
	Available         bool
	DeviceName        string
	TotalMemory       uint64
	FreeMemory        uint64
	SMVersion         string
	ComputeCapability string
}

// GetCUDAInfo returns information about the CUDA device.
func GetCUDAInfo() CUDAInfo {
	if !torch.IsCUDA() {
		return CUDAInfo{Available: false}
	}

	smVersion := torch.GetCUDASMVersion()

	return CUDAInfo{
		Available:         true,
		DeviceName:        torch.GetCUDADeviceName(),
		TotalMemory:       torch.GetCUDAMemoryTotal(),
		FreeMemory:        torch.GetCUDAMemoryFree(),
		SMVersion:         smVersion,
		ComputeCapability: smVersion, // Same as SM version.
	}
}

// MemoryUsagePercent returns the percentage of GPU memory currently in use.
func MemoryUsagePercent() float64 {
	total := MemoryTotal()
	if total == 0 {
		return 0
	}

	allocated := MemoryAllocated()

	return float64(allocated) / float64(total) * 100.0
}

// Synchronize synchronizes the CUDA device.
// This waits for all CUDA kernels to complete.
func Synchronize() {
	if torch.IsCUDA() {
		torch.CUDASynchronize()
	}
}

// EmptyCache releases all unused cached memory from the CUDA allocator.
func EmptyCache() {
	if torch.IsCUDA() {
		torch.CUDAEmptyCache()
	}
}

// MemoryAllocated returns the current GPU memory usage in bytes.
func MemoryAllocated() uint64 {
	if !torch.IsCUDA() {
		return 0
	}

	return torch.GetCUDAMemoryAllocated()
}

// MemoryTotal returns the total GPU memory in bytes.
func MemoryTotal() uint64 {
	if !torch.IsCUDA() {
		return 0
	}

	return torch.GetCUDAMemoryTotal()
}
