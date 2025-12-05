package nn

import (
	"fmt"
	"testing"

	"github.com/empyrealm/gotorch/consts"
	"github.com/empyrealm/gotorch/tensor"
)

func TestLayerNorm(t *testing.T) {
	l := NewLayerNorm(2)
	x := tensor.ARange(4, consts.KFloat).View(2, 2)
	y := l.Forward(x)
	fmt.Println(y.Float32Value())
	for _, p := range l.Parameters() {
		fmt.Println(p.Float32Value())
	}
}
