package optimizer

import (
	"math"
)

// ============================================================================
// Learning Rate Schedulers
// ============================================================================

// Scheduler is the interface for learning rate schedulers.
type Scheduler interface {
	// Step advances the scheduler by one step and returns the new LR.
	Step() float64
	// GetLR returns the current learning rate.
	GetLR() float64
	// SetLR manually sets the learning rate.
	SetLR(lr float64)
}

// ============================================================================
// Cosine Annealing Scheduler
// ============================================================================

// CosineAnnealingLR implements cosine annealing with warm restarts.
type CosineAnnealingLR struct {
	optimizer Optimizer
	baseLR    float64
	minLR     float64
	tMax      int     // Period length
	tMult     float64 // Period multiplier after restart
	step      int
	currentLR float64
	cycle     int
}

// NewCosineAnnealingLR creates a cosine annealing scheduler.
func NewCosineAnnealingLR(opt Optimizer, baseLR, minLR float64, tMax int, tMult float64) *CosineAnnealingLR {
	return &CosineAnnealingLR{
		optimizer: opt,
		baseLR:    baseLR,
		minLR:     minLR,
		tMax:      tMax,
		tMult:     tMult,
		step:      0,
		currentLR: baseLR,
		cycle:     0,
	}
}

// Step advances the scheduler.
func (s *CosineAnnealingLR) Step() float64 {
	// Calculate current period length.
	periodLen := float64(s.tMax) * math.Pow(s.tMult, float64(s.cycle))

	// Position within current period.
	t := float64(s.step) / periodLen

	// Cosine annealing.
	s.currentLR = s.minLR + 0.5*(s.baseLR-s.minLR)*(1+math.Cos(math.Pi*t))

	s.step++

	// Check for restart.
	if float64(s.step) >= periodLen {
		s.step = 0
		s.cycle++
	}

	// Update optimizer LR.
	s.optimizer.SetLr(s.currentLR)

	return s.currentLR
}

// GetLR returns current learning rate.
func (s *CosineAnnealingLR) GetLR() float64 {
	return s.currentLR
}

// SetLR manually sets LR.
func (s *CosineAnnealingLR) SetLR(lr float64) {
	s.currentLR = lr
	s.optimizer.SetLr(lr)
}

// ============================================================================
// Linear Warmup Scheduler
// ============================================================================

// LinearWarmup implements linear warmup then constant LR.
type LinearWarmup struct {
	optimizer   Optimizer
	baseLR      float64
	warmupSteps int
	step        int
	currentLR   float64
}

// NewLinearWarmup creates a linear warmup scheduler.
func NewLinearWarmup(opt Optimizer, baseLR float64, warmupSteps int) *LinearWarmup {
	startLR := baseLR / 10.0
	opt.SetLr(startLR)
	return &LinearWarmup{
		optimizer:   opt,
		baseLR:      baseLR,
		warmupSteps: warmupSteps,
		step:        0,
		currentLR:   startLR,
	}
}

// Step advances the scheduler.
func (s *LinearWarmup) Step() float64 {
	if s.step < s.warmupSteps {
		// Linear warmup.
		progress := float64(s.step) / float64(s.warmupSteps)
		s.currentLR = (s.baseLR / 10.0) + progress*(s.baseLR-s.baseLR/10.0)
	} else {
		s.currentLR = s.baseLR
	}

	s.step++
	s.optimizer.SetLr(s.currentLR)

	return s.currentLR
}

// GetLR returns current learning rate.
func (s *LinearWarmup) GetLR() float64 {
	return s.currentLR
}

// SetLR manually sets LR.
func (s *LinearWarmup) SetLR(lr float64) {
	s.currentLR = lr
	s.optimizer.SetLr(lr)
}

// ============================================================================
// Warmup + Cosine Decay
// ============================================================================

// WarmupCosineDecay combines linear warmup with cosine decay.
type WarmupCosineDecay struct {
	optimizer   Optimizer
	baseLR      float64
	minLR       float64
	warmupSteps int
	totalSteps  int
	step        int
	currentLR   float64
}

// NewWarmupCosineDecay creates a warmup + cosine decay scheduler.
func NewWarmupCosineDecay(opt Optimizer, baseLR, minLR float64, warmupSteps, totalSteps int) *WarmupCosineDecay {
	startLR := baseLR / 10.0
	opt.SetLr(startLR)
	return &WarmupCosineDecay{
		optimizer:   opt,
		baseLR:      baseLR,
		minLR:       minLR,
		warmupSteps: warmupSteps,
		totalSteps:  totalSteps,
		step:        0,
		currentLR:   startLR,
	}
}

// Step advances the scheduler.
func (s *WarmupCosineDecay) Step() float64 {
	if s.step < s.warmupSteps {
		// Linear warmup.
		progress := float64(s.step) / float64(s.warmupSteps)
		s.currentLR = (s.baseLR / 10.0) + progress*(s.baseLR-s.baseLR/10.0)
	} else {
		// Cosine decay.
		progress := float64(s.step-s.warmupSteps) / float64(s.totalSteps-s.warmupSteps)
		progress = math.Min(progress, 1.0)
		s.currentLR = s.minLR + 0.5*(s.baseLR-s.minLR)*(1+math.Cos(math.Pi*progress))
	}

	s.step++
	s.optimizer.SetLr(s.currentLR)

	return s.currentLR
}

// GetLR returns current learning rate.
func (s *WarmupCosineDecay) GetLR() float64 {
	return s.currentLR
}

// SetLR manually sets LR.
func (s *WarmupCosineDecay) SetLR(lr float64) {
	s.currentLR = lr
	s.optimizer.SetLr(lr)
}

// ============================================================================
// Exponential Decay
// ============================================================================

// ExponentialDecay implements exponential LR decay.
type ExponentialDecay struct {
	optimizer Optimizer
	baseLR    float64
	gamma     float64 // Decay factor per step
	step      int
	currentLR float64
}

// NewExponentialDecay creates an exponential decay scheduler.
func NewExponentialDecay(opt Optimizer, baseLR, gamma float64) *ExponentialDecay {
	opt.SetLr(baseLR)
	return &ExponentialDecay{
		optimizer: opt,
		baseLR:    baseLR,
		gamma:     gamma,
		step:      0,
		currentLR: baseLR,
	}
}

// Step advances the scheduler.
func (s *ExponentialDecay) Step() float64 {
	s.currentLR = s.baseLR * math.Pow(s.gamma, float64(s.step))
	s.step++
	s.optimizer.SetLr(s.currentLR)
	return s.currentLR
}

// GetLR returns current learning rate.
func (s *ExponentialDecay) GetLR() float64 {
	return s.currentLR
}

// SetLR manually sets LR.
func (s *ExponentialDecay) SetLR(lr float64) {
	s.currentLR = lr
	s.optimizer.SetLr(lr)
}
