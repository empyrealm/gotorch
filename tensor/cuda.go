package tensor

import (
	"fmt"

	"github.com/empyrealm/gotorch/internal/torch"
)


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


// MemoryTotal returns the total CUDA memory in bytes.
func MemoryTotal() int64 {
	return torch.CUDAMemoryTotal()
}


// MemoryFree returns the free CUDA memory in bytes.
func MemoryFree() int64 {
	return torch.CUDAMemoryFree()
}


// DeviceCount returns the number of CUDA devices.
func DeviceCount() int {
	return torch.CUDADeviceCount()
}


// DeviceName returns the name of a CUDA device.
func DeviceName(deviceID int) string {
	return torch.CUDADeviceName(deviceID)
}


// ComputeCapability returns the compute capability (e.g., 89 for sm_89).
func ComputeCapability(deviceID int) int {
	return torch.CUDAComputeCapability(deviceID)
}


// CUDAInfo holds information about CUDA device and memory.
type CUDAInfo struct {
	DeviceName        string
	DeviceCount       int
	ComputeCapability int
	TotalMemory       int64
	FreeMemory        int64
	AllocatedMemory   int64
	ReservedMemory    int64
}


// GetCUDAInfo returns comprehensive CUDA information.
func GetCUDAInfo() CUDAInfo {
	return CUDAInfo{
		DeviceName:        DeviceName(0),
		DeviceCount:       DeviceCount(),
		ComputeCapability: ComputeCapability(0),
		TotalMemory:       MemoryTotal(),
		FreeMemory:        MemoryFree(),
		AllocatedMemory:   MemoryAllocated(),
		ReservedMemory:    MemoryReserved(),
	}
}


// String returns a formatted string of CUDA info.
func (c CUDAInfo) String() string {
	return fmt.Sprintf(`CUDA Device: %s (SM %d.%d)
  Total Memory:     %.2f GB
  Free Memory:      %.2f GB
  Allocated:        %.2f GB
  Reserved:         %.2f GB`,
		c.DeviceName,
		c.ComputeCapability/10, c.ComputeCapability%10,
		float64(c.TotalMemory)/1e9,
		float64(c.FreeMemory)/1e9,
		float64(c.AllocatedMemory)/1e9,
		float64(c.ReservedMemory)/1e9,
	)
}


// MemoryUsagePercent returns current memory usage as percentage.
func MemoryUsagePercent() float64 {
	total := MemoryTotal()
	if total == 0 {
		return 0
	}
	return float64(MemoryAllocated()) / float64(total) * 100
}
