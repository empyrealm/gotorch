package init

import (
	"github.com/empyrealm/gotorch/internal/torch"
	"github.com/empyrealm/gotorch/tensor"
)

func KaimingUniform(t *tensor.Tensor, a float64) {
	torch.InitKaimingUniform(t.Tensor(), a)
}

func XaiverUniform(t *tensor.Tensor, gain float64) {
	torch.InitXaiverUniform(t.Tensor(), gain)
}

func Normal(t *tensor.Tensor, mean, std float64) {
	torch.InitNormal(t.Tensor(), mean, std)
}

func Zeros(t *tensor.Tensor) {
	torch.InitZeros(t.Tensor())
}
