// Package tensor provides TensorScope for explicit memory management.
package tensor

import (
	"sync"
)

// Scope tracks tensors for explicit memory management.
// Use NewScope() to create a scope, Track() tensors, and Close() to free them.
type Scope struct {
	mu      sync.Mutex
	tensors []*Tensor
	kept    map[*Tensor]bool
}

// NewScope creates a new tensor scope for tracking tensors.
func NewScope() *Scope {
	return &Scope{
		tensors: make([]*Tensor, 0, 32),
		kept:    make(map[*Tensor]bool),
	}
}

// Track adds a tensor to be freed when the scope closes.
func (s *Scope) Track(t *Tensor) {
	if t == nil || t.t == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.tensors = append(s.tensors, t)
}

// TrackAll adds multiple tensors to be freed when the scope closes.
func (s *Scope) TrackAll(tensors ...*Tensor) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, t := range tensors {
		if t != nil && t.t != nil {
			s.tensors = append(s.tensors, t)
		}
	}
}

// Keep marks a tensor to NOT be freed when the scope closes.
// Use this for tensors that should outlive the scope.
func (s *Scope) Keep(t *Tensor) {
	if t == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.kept[t] = true
}

// KeepAll marks multiple tensors to NOT be freed when the scope closes.
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
func (s *Scope) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, t := range s.tensors {
		if t == nil || t.t == nil {
			continue
		}

		// Skip tensors marked as kept.
		if s.kept[t] {
			continue
		}

		t.Free()
	}

	// Clear the slices.
	s.tensors = nil
	s.kept = nil
}
