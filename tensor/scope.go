// tensor/scope.go
// TensorScope provides explicit memory management for CUDA tensors.
//
// Go's garbage collector doesn't see CUDA memory pressure, so it doesn't
// free GPU tensors fast enough. TensorScope tracks tensors created within
// a scope and frees them when the scope ends.
//
// Usage:
//
//	scope := tensor.NewScope()
//	defer scope.Close()
//
//	// Tensors created while scope is active are tracked.
//	a := tensor.Rand([]int64{100, 100}, consts.KCUDA)
//	b := a.Add(a)
//	c := b.Mul(a)
//
//	// Keep tensors you need.
//	scope.Keep(c)
//
//	// When scope.Close() is called, a and b are freed, but c is kept.
package tensor

import (
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/lwch/logging"
)

// scopeRegistry maps goroutine IDs to their active scope.
var scopeRegistry sync.Map

var scopeCounter uint64
var scopeCounterMu sync.Mutex

// Scope tracks tensors for explicit cleanup.
type Scope struct {
	id       uint64
	tensors  []*Tensor
	kept     map[*Tensor]bool
	mu       sync.Mutex
	parentID uint64
}

func nextScopeID() uint64 {
	scopeCounterMu.Lock()
	defer scopeCounterMu.Unlock()
	scopeCounter++
	return scopeCounter
}

// NewScope creates a new tensor scope.
// Tensors created while a scope is active are automatically tracked.
func NewScope() *Scope {

	s := &Scope{
		id:      nextScopeID(),
		tensors: make([]*Tensor, 0, 100),
		kept:    make(map[*Tensor]bool),
	}

	// Get current goroutine's scope (if any) as parent.
	goroutineID := getGoroutineID()
	if existing, ok := scopeRegistry.Load(goroutineID); ok {
		s.parentID = existing.(*Scope).id
	}

	// Register this scope for the current goroutine.
	scopeRegistry.Store(goroutineID, s)

	logging.Debug("tensor scope %d created", s.id)

	return s
}

// Track adds a tensor to this scope for cleanup.
// Called automatically by tensor.New() when a scope is active.
func (s *Scope) Track(t *Tensor) {
	if t == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.tensors = append(s.tensors, t)
}

// Keep marks a tensor to be kept (not freed) when the scope closes.
// Use this for tensors you need to return or use after the scope.
func (s *Scope) Keep(t *Tensor) {
	if t == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.kept[t] = true
}

// KeepAll marks multiple tensors to be kept.
func (s *Scope) KeepAll(tensors ...*Tensor) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, t := range tensors {
		if t != nil {
			s.kept[t] = true
		}
	}
}

// Close frees all tracked tensors except those marked as kept.
// Always call this (preferably via defer) to prevent memory leaks.
func (s *Scope) Close() {

	s.mu.Lock()
	defer s.mu.Unlock()

	freed := 0
	kept := 0

	for _, t := range s.tensors {
		if t == nil || t.t == nil {
			continue
		}

		if s.kept[t] {
			kept++
			continue
		}

		// Free the tensor.
		t.Free()
		freed++
	}

	logging.Debug("tensor scope %d closed: freed %d, kept %d", s.id, freed, kept)

	// Clear the slice.
	s.tensors = nil
	s.kept = nil

	// Restore parent scope (if any).
	goroutineID := getGoroutineID()
	if s.parentID != 0 {
		// Find parent scope and restore it.
		scopeRegistry.Range(func(key, value interface{}) bool {
			if scope, ok := value.(*Scope); ok && scope.id == s.parentID {
				scopeRegistry.Store(goroutineID, scope)
				return false
			}
			return true
		})
	} else {
		scopeRegistry.Delete(goroutineID)
	}
}

// Count returns the number of tracked tensors.
func (s *Scope) Count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.tensors)
}

// GetActiveScope returns the active scope for the current goroutine, if any.
func GetActiveScope() *Scope {
	goroutineID := getGoroutineID()
	if scope, ok := scopeRegistry.Load(goroutineID); ok {
		return scope.(*Scope)
	}
	return nil
}

// getGoroutineID returns a unique ID for the current goroutine.
func getGoroutineID() uint64 {
	var buf [64]byte
	n := runtime.Stack(buf[:], false)
	// Stack trace starts with "goroutine <id> ["
	idField := strings.Fields(strings.TrimPrefix(string(buf[:n]), "goroutine "))[0]
	id, _ := strconv.ParseUint(idField, 10, 64)
	return id
}
