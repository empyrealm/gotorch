package torch

// #include "tensor.h"
import "C"

import "github.com/empyrealm/gotorch/consts"

// AutocastSetEnabled enables or disables autocast for CUDA.
func AutocastSetEnabled(enabled bool) {
	C.autocast_set_enabled(C.bool(enabled))
}

// AutocastIsEnabled returns true if autocast is enabled.
func AutocastIsEnabled() bool {
	return bool(C.autocast_is_enabled())
}

// AutocastSetDtype sets the autocast target dtype (e.g., Half, BFloat16).
func AutocastSetDtype(dtype consts.ScalarType) {
	C.autocast_set_dtype(C.int8_t(dtype))
}

// AutocastGetDtype returns the current autocast dtype.
func AutocastGetDtype() consts.ScalarType {
	return consts.ScalarType(C.autocast_get_dtype())
}

// AutocastClearCache clears the autocast weight cache.
func AutocastClearCache() {
	C.autocast_clear_cache()
}

// AutocastIncrementNesting increments the autocast nesting level.
func AutocastIncrementNesting() {
	C.autocast_increment_nesting()
}

// AutocastDecrementNesting decrements the autocast nesting level.
func AutocastDecrementNesting() {
	C.autocast_decrement_nesting()
}
