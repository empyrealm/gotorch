package nn

import (
	"github.com/empyrealm/gotorch/internal/torch"
)

type Embedding struct {
	module
}

func NewEmbedding(numEmbeddings, embeddingDim, paddingIdx int64) *Embedding {
	return &Embedding{
		module{torch.NewEmbedding(numEmbeddings, embeddingDim, paddingIdx)},
	}
}
