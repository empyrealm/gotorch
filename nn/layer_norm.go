package nn

import (
	"github.com/empyrealm/gotorch/internal/torch"
)

type LayerNorm struct {
	module
}

func NewLayerNorm(shapes ...int64) *LayerNorm {
	return &LayerNorm{
		module{torch.NewLayerNorm(shapes)},
	}
}
