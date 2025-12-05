package nn

import (
	"github.com/empyrealm/gotorch/consts"
	"github.com/empyrealm/gotorch/internal/torch"
	"github.com/empyrealm/gotorch/tensor"
)

type module struct {
	m torch.Module
}

func (m *module) Parameters() []*tensor.Tensor {
	params := m.m.Parameters()
	ret := make([]*tensor.Tensor, len(params))
	for i, p := range params {
		ret[i] = tensor.New(p)
	}
	return ret
}

func (m *module) Forward(x *tensor.Tensor) *tensor.Tensor {
	t := m.m.(torch.NormalForward).Forward(x.Tensor())
	return tensor.New(t)
}

func (m *module) ToDevice(device consts.DeviceType) {
	m.m.ToDevice(device)
}

func (m *module) ToScalarType(t consts.ScalarType) {
	m.m.ToScalarType(t)
}
