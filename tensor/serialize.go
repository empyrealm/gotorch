package tensor

import (
	"github.com/empyrealm/gotorch/internal/torch"
)

// SavePT saves tensors to a .pt file (PyTorch format).
// This is the native PyTorch serialization format, compatible with
// torch.load() in Python.
//
// Example:
//
//	params := model.Parameters()
//	tensor.SavePT(params, "model.pt")
func SavePT(tensors []*Tensor, path string) {
	if len(tensors) == 0 {
		return
	}

	internal := make([]torch.Tensor, len(tensors))
	for i, t := range tensors {
		internal[i] = t.t
	}

	torch.SaveTensors(internal, path)
}

// LoadPT loads tensors from a .pt file (PyTorch format).
// Returns a slice of tensors in the same order they were saved.
//
// Example:
//
//	params := tensor.LoadPT("model.pt")
//	model.SetParameters(params)
func LoadPT(path string) []*Tensor {
	internal := torch.LoadTensors(path)
	if len(internal) == 0 {
		return nil
	}

	tensors := make([]*Tensor, len(internal))
	for i, t := range internal {
		tensors[i] = New(t)
	}

	return tensors
}

