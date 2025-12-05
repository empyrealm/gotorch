package torch

// #include "tensor.h"
import "C"

// ============================================================================
// Vectorized Trading Environment Operations
// ============================================================================

// EnvUpdatePositions updates positions based on actions.
func EnvUpdatePositions(positions, actions Tensor) Tensor {
	var err *C.char
	ret := C.env_update_positions(&err, C.tensor(positions), C.tensor(actions))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// EnvCalculatePnL calculates profit/loss.
func EnvCalculatePnL(positions, currentPrices, entryPrices Tensor) Tensor {
	var err *C.char
	ret := C.env_calculate_pnl(&err, C.tensor(positions), C.tensor(currentPrices), C.tensor(entryPrices))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// EnvCalculateFees calculates trading fees.
func EnvCalculateFees(volumes Tensor, feeRate float64) Tensor {
	var err *C.char
	ret := C.env_calculate_fees(&err, C.tensor(volumes), C.double(feeRate))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// EnvCalculateRewards calculates rewards with risk adjustment.
func EnvCalculateRewards(pnl, fees Tensor, drawdownPenalty float64, maxEquity Tensor) Tensor {
	var err *C.char
	ret := C.env_calculate_rewards(&err, C.tensor(pnl), C.tensor(fees), C.double(drawdownPenalty), C.tensor(maxEquity))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// EnvCheckDone checks if environments are done.
func EnvCheckDone(steps Tensor, maxSteps int64, equity Tensor, minEquity float64) Tensor {
	var err *C.char
	ret := C.env_check_done(&err, C.tensor(steps), C.int64_t(maxSteps), C.tensor(equity), C.double(minEquity))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// EnvBuildState constructs observation state.
func EnvBuildState(marketFeatures, positions, equity Tensor) Tensor {
	var err *C.char
	ret := C.env_build_state(&err, C.tensor(marketFeatures), C.tensor(positions), C.tensor(equity))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ret)
}

// EnvVectorizedStep performs a full environment step on GPU.
func EnvVectorizedStep(
	marketData Tensor,
	stepIndices Tensor,
	positions Tensor,
	entryPrices Tensor,
	equity Tensor,
	maxEquity Tensor,
	actions Tensor,
	feeRate float64,
	minEquity float64,
	maxSteps int64,
	outStates Tensor,
	outRewards Tensor,
	outDones Tensor,
	outPositions Tensor,
	outEntryPrices Tensor,
	outEquity Tensor,
	outMaxEquity Tensor,
	outStepIndices Tensor,
) {
	var err *C.char
	C.env_vectorized_step(
		&err,
		C.tensor(marketData),
		C.tensor(stepIndices),
		C.tensor(positions),
		C.tensor(entryPrices),
		C.tensor(equity),
		C.tensor(maxEquity),
		C.tensor(actions),
		C.double(feeRate),
		C.double(minEquity),
		C.int64_t(maxSteps),
		C.tensor(outStates),
		C.tensor(outRewards),
		C.tensor(outDones),
		C.tensor(outPositions),
		C.tensor(outEntryPrices),
		C.tensor(outEquity),
		C.tensor(outMaxEquity),
		C.tensor(outStepIndices),
	)
	if err != nil {
		panic(C.GoString(err))
	}
}

// EnvResetDone resets environments that are done.
func EnvResetDone(
	dones Tensor,
	positions Tensor,
	entryPrices Tensor,
	equity Tensor,
	maxEquity Tensor,
	stepIndices Tensor,
	initialEquity float64,
) {
	var err *C.char
	C.env_reset_done(
		&err,
		C.tensor(dones),
		C.tensor(positions),
		C.tensor(entryPrices),
		C.tensor(equity),
		C.tensor(maxEquity),
		C.tensor(stepIndices),
		C.double(initialEquity),
	)
	if err != nil {
		panic(C.GoString(err))
	}
}
