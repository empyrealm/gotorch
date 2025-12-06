#include "tensor.h"
#include "exception.hpp"
#include <torch/torch.h>

tensor new_tensor(char **err)
{
    return auto_catch_tensor([]() { return new torch::Tensor(); }, err);
}

void free_tensor(tensor t)
{
    delete t;
}

tensor tensor_to_device(char **err, tensor t, int8_t device)
{
    return auto_catch_tensor([t, device]() { return new torch::Tensor(t->to(torch::DeviceType(device))); }, err);
}

tensor tensor_to_scalar_type(char **err, tensor t, int8_t scalar_type)
{
    return auto_catch_tensor([t, scalar_type]() { return new torch::Tensor(t->to(torch::ScalarType(scalar_type))); },
                             err);
}

tensor tensor_clone(char **err, tensor t)
{
    return auto_catch_tensor([t]() { return new torch::Tensor(t->clone()); }, err);
}

tensor tensor_arange(char **err, int end, int8_t dtype, int8_t device)
{
    return auto_catch_tensor(
        [end, dtype, device]() {
            return new torch::Tensor(torch::arange(
                end, torch::TensorOptions().dtype(torch::ScalarType(dtype)).device(torch::DeviceType(device))));
        },
        err);
}

tensor tensor_zeros(char **err, int64_t *shape, size_t shape_len, int8_t dtype, int8_t device)
{
    return auto_catch_tensor(
        [shape, shape_len, dtype, device]() {
            return new torch::Tensor(
                torch::zeros(torch::IntArrayRef(shape, shape_len),
                             torch::TensorOptions().dtype(torch::ScalarType(dtype)).device(torch::DeviceType(device))));
        },
        err);
}

tensor tensor_from_data(char **err, void *data, int64_t *shape, size_t shape_len, int8_t dtype, int8_t device)
{
    return auto_catch_tensor(
        [data, shape, shape_len, dtype, device]() {
            torch::Tensor zeros = torch::zeros(torch::IntArrayRef(shape, shape_len),
                                               torch::TensorOptions().dtype(torch::ScalarType(dtype)));
            memcpy(zeros.data_ptr(), data, zeros.numel() * zeros.element_size());
            return new torch::Tensor(zeros.to(torch::DeviceType(device)));
        },
        err);
}

void tensor_copy_data(tensor t, void *data)
{
    // CRITICAL: For CUDA tensors, copy to CPU first before accessing data_ptr.
    // Directly accessing data_ptr on CUDA tensors from CPU code causes segfaults.
    if (t->device().is_cuda()) {
        torch::Tensor cpu_tensor = t->to(torch::kCPU);
        memcpy(data, cpu_tensor.data_ptr(), cpu_tensor.numel() * cpu_tensor.element_size());
    } else {
    memcpy(data, t->data_ptr(), t->numel() * t->element_size());
    }
}

// tensor_set_data copies data FROM buffer INTO tensor (reverse of tensor_copy_data).
// This enables loading model weights from saved checkpoints.
void tensor_set_data(char **err, tensor t, void *data)
{
    return auto_catch_void([t, data]() {
        // Use no_grad to allow in-place modification of leaf variables.
        torch::NoGradGuard no_grad;
        
        // For CUDA tensors, we need to copy to CPU first, then back.
        if (t->device().is_cuda()) {
            // Create CPU tensor with same shape and copy data into it.
            torch::Tensor cpu_tensor = torch::empty_like(*t, torch::TensorOptions().device(torch::kCPU));
            memcpy(cpu_tensor.data_ptr(), data, cpu_tensor.numel() * cpu_tensor.element_size());
            // Copy back to original CUDA tensor.
            t->copy_(cpu_tensor);
        } else {
            // Direct memcpy for CPU tensors.
            memcpy(t->data_ptr(), data, t->numel() * t->element_size());
        }
    }, err);
}

void tensor_set_requires_grad(char **err, tensor t, bool b)
{
    return auto_catch_void([t, b]() { t->set_requires_grad(b); }, err);
}

size_t tensor_elem_size(tensor t)
{
    return t->element_size();
}

size_t tensor_elem_count(tensor t)
{
    return t->numel();
}

int tensor_scalar_type(tensor t)
{
    return int(t->scalar_type());
}

size_t tensor_dims(tensor t)
{
    return t->dim();
}

void tensor_shapes(tensor t, int64_t *shapes)
{
    size_t dim = t->dim();
    for (size_t i = 0; i < dim; i++)
    {
        shapes[i] = t->size(i);
    }
}

int8_t tensor_device_type(tensor t)
{
    return int8_t(t->device().type());
}

tensor tensor_reshape(char **err, tensor t, int64_t *shape, size_t shape_len)
{
    return auto_catch_tensor(
        [t, shape, shape_len]() { return new torch::Tensor(t->reshape(torch::IntArrayRef(shape, shape_len))); }, err);
}

tensor tensor_transpose(char **err, tensor t, int64_t dim1, int64_t dim2)
{
    return auto_catch_tensor([t, dim1, dim2]() { return new torch::Tensor(t->transpose(dim1, dim2)); }, err);
}

tensor tensor_narrow(char **err, tensor t, int64_t dim, int64_t start, int64_t length)
{
    return auto_catch_tensor([t, dim, start, length]() { return new torch::Tensor(t->narrow(dim, start, length)); },
                             err);
}

tensor tensor_vstack(char **err, tensor a, tensor b)
{
    return auto_catch_tensor([a, b]() { return new torch::Tensor(torch::vstack(torch::TensorList({*a, *b}))); }, err);
}

tensor tensor_hstack(char **err, tensor a, tensor b)
{
    return auto_catch_tensor([a, b]() { return new torch::Tensor(torch::hstack(torch::TensorList({*a, *b}))); }, err);
}

tensor tensor_view(char **err, tensor a, int64_t *shape, size_t shape_len)
{
    return auto_catch_tensor(
        [a, shape, shape_len]() { return new torch::Tensor(a->view(torch::IntArrayRef(shape, shape_len))); }, err);
}

tensor tensor_permute(char **err, tensor a, int64_t *dims, size_t dims_len)
{
    return auto_catch_tensor(
        [a, dims, dims_len]() { return new torch::Tensor(a->permute(torch::IntArrayRef(dims, dims_len))); }, err);
}

tensor tensor_detach(tensor t)
{
    return new torch::Tensor(t->detach());
}

// ============================================================================
// Model Serialization (.pt format)
// ============================================================================

void tensors_save(char **err, tensor *tensors, size_t count, const char *path)
{
    return auto_catch_void([tensors, count, path]()
                           {
                               std::vector<torch::Tensor> tensor_list;
                               tensor_list.reserve(count);
                               for (size_t i = 0; i < count; i++) {
                                   tensor_list.push_back(*tensors[i]);
                               }
                               torch::save(tensor_list, path);
                           },
                           err);
}

size_t tensors_load(char **err, const char *path, tensor **out_tensors)
{
    std::vector<torch::Tensor> tensor_list;
    
    try {
        torch::load(tensor_list, path);
    } catch (const std::exception &e) {
        *err = strdup(e.what());
        return 0;
    }
    
    // Allocate output array.
    *out_tensors = (tensor *)malloc(tensor_list.size() * sizeof(tensor));
    for (size_t i = 0; i < tensor_list.size(); i++) {
        (*out_tensors)[i] = new torch::Tensor(tensor_list[i]);
    }
    
    return tensor_list.size();
}

void tensors_free_array(tensor *tensors, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        delete tensors[i];
    }
    free(tensors);
}

// ============================================================================
// GPU-side Sampling Operations (avoid CPU transfer)
// ============================================================================

// Multinomial sampling on GPU - critical for RL action selection.
tensor tensor_multinomial(char **err, tensor probs, int64_t num_samples, bool replacement)
{
    return auto_catch_tensor([probs, num_samples, replacement]()
                             { return new torch::Tensor(torch::multinomial(*probs, num_samples, replacement)); },
                             err);
}

// Categorical sampling (single sample from each row).
tensor tensor_categorical_sample(char **err, tensor logits)
{
    return auto_catch_tensor([logits]()
                             {
                                 // Softmax then sample.
                                 auto probs = torch::softmax(*logits, -1);
                                 return new torch::Tensor(torch::multinomial(probs, 1, false).squeeze(-1));
                             },
                             err);
}

// Sample from normal distribution: sample = mean + std * randn_like(mean).
tensor tensor_normal_sample(char **err, tensor mean, tensor std)
{
    return auto_catch_tensor([mean, std]()
                             {
                                 auto noise = torch::randn_like(*mean);
                                 return new torch::Tensor(*mean + *std * noise);
                             },
                             err);
}

// Argmax on GPU (for greedy action selection).
tensor tensor_argmax(char **err, tensor t, int64_t dim, bool keepdim)
{
    return auto_catch_tensor([t, dim, keepdim]()
                             { return new torch::Tensor(torch::argmax(*t, dim, keepdim)); },
                             err);
}

// Random uniform tensor on GPU.
tensor tensor_rand(char **err, int64_t *shape, size_t shape_len, int8_t device)
{
    return auto_catch_tensor([shape, shape_len, device]()
                             {
                                 return new torch::Tensor(torch::rand(
                                     torch::IntArrayRef(shape, shape_len),
                                     torch::TensorOptions().device(torch::DeviceType(device))));
                             },
                             err);
}

// Random normal tensor on GPU.
tensor tensor_randn(char **err, int64_t *shape, size_t shape_len, int8_t device)
{
    return auto_catch_tensor([shape, shape_len, device]()
                             {
                                 return new torch::Tensor(torch::randn(
                                     torch::IntArrayRef(shape, shape_len),
                                     torch::TensorOptions().device(torch::DeviceType(device))));
                             },
                             err);
}

// Clamp tensor values (for action bounds).
tensor tensor_clamp_minmax(char **err, tensor t, double min_val, double max_val)
{
    return auto_catch_tensor([t, min_val, max_val]()
                             { return new torch::Tensor(torch::clamp(*t, min_val, max_val)); },
                             err);
}

// Where operation (conditional selection on GPU).
tensor tensor_where(char **err, tensor condition, tensor x, tensor y)
{
    return auto_catch_tensor([condition, x, y]()
                             { return new torch::Tensor(torch::where(*condition, *x, *y)); },
                             err);
}

// ============================================================================
// Indexing Operations (for replay buffers)
// ============================================================================

// Index read: t[idx0, idx1, ...]
tensor tensor_index(char **err, tensor t, int64_t *indices, size_t indices_len)
{
    return auto_catch_tensor([t, indices, indices_len]()
                             {
                                 std::vector<at::indexing::TensorIndex> idx;
                                 for (size_t i = 0; i < indices_len; i++)
                                 {
                                     idx.push_back(indices[i]);
                                 }
                                 return new torch::Tensor(t->index(idx));
                             },
                             err);
}

// Index write: t[idx] = value
void tensor_index_put(char **err, tensor t, int64_t *indices, size_t indices_len, tensor value)
{
    return auto_catch_void([t, indices, indices_len, value]()
                           {
                               std::vector<at::indexing::TensorIndex> idx;
                               for (size_t i = 0; i < indices_len; i++)
                               {
                                   idx.push_back(indices[i]);
                               }
                               t->index_put_(idx, *value);
                           },
                           err);
}

// Index write with tensor indices: t[indices_tensor] = value
void tensor_index_put_tensor(char **err, tensor t, tensor indices_tensor, tensor value)
{
    return auto_catch_void([t, indices_tensor, value]()
                           { t->index_put_({*indices_tensor}, *value); },
                           err);
}

// Index select: t.index_select(dim, indices)
tensor tensor_index_select(char **err, tensor t, int64_t dim, tensor indices)
{
    return auto_catch_tensor([t, dim, indices]()
                             { return new torch::Tensor(t->index_select(dim, *indices)); },
                             err);
}

// ============================================================================
// Global Reduction Operations (over all elements)
// ============================================================================

tensor tensor_mean_all(char **err, tensor t)
{
    return auto_catch_tensor([t]()
                             { return new torch::Tensor(t->mean()); },
                             err);
}

tensor tensor_sum_all(char **err, tensor t)
{
    return auto_catch_tensor([t]()
                             { return new torch::Tensor(t->sum()); },
                             err);
}

tensor tensor_max_all(char **err, tensor t)
{
    return auto_catch_tensor([t]()
                             { return new torch::Tensor(t->max()); },
                             err);
}

tensor tensor_min_all(char **err, tensor t)
{
    return auto_catch_tensor([t]()
                             { return new torch::Tensor(t->min()); },
                             err);
}

tensor tensor_std_all(char **err, tensor t, bool unbiased)
{
    return auto_catch_tensor([t, unbiased]()
                             { return new torch::Tensor(t->std(unbiased)); },
                             err);
}

// Power with tensor exponent
tensor tensor_pow_tensor(char **err, tensor t, tensor exp)
{
    return auto_catch_tensor([t, exp]()
                             { return new torch::Tensor(t->pow(*exp)); },
                             err);
}

// Ones tensor
tensor tensor_ones(char **err, int64_t *shape, size_t shape_len, int8_t dtype, int8_t device)
{
    return auto_catch_tensor([shape, shape_len, dtype, device]()
                             {
                                 return new torch::Tensor(torch::ones(
                                     torch::IntArrayRef(shape, shape_len),
                                     torch::TensorOptions()
                                         .dtype(torch::ScalarType(dtype))
                                         .device(torch::DeviceType(device))));
                             },
                             err);
}

// ============================================================================
// Mixed Precision Support (AMP - Automatic Mixed Precision)
// ============================================================================

// Convert to half precision (fp16).
tensor tensor_half(char **err, tensor t)
{
    return auto_catch_tensor([t]()
                             { return new torch::Tensor(t->to(torch::kHalf)); },
                             err);
}

// Convert to bfloat16.
tensor tensor_bfloat16(char **err, tensor t)
{
    return auto_catch_tensor([t]()
                             { return new torch::Tensor(t->to(torch::kBFloat16)); },
                             err);
}

// Convert to float32.
tensor tensor_float32(char **err, tensor t)
{
    return auto_catch_tensor([t]()
                             { return new torch::Tensor(t->to(torch::kFloat32)); },
                             err);
}

// Check if tensor is half precision.
bool tensor_is_half(tensor t)
{
    return t->scalar_type() == torch::kHalf;
}

// Check if tensor is bfloat16.
bool tensor_is_bfloat16(tensor t)
{
    return t->scalar_type() == torch::kBFloat16;
}

// ============================================================================
// Gradient Scaling for Mixed Precision
// ============================================================================

// Scale tensor (for gradient scaling in AMP).
tensor tensor_scale(char **err, tensor t, double scale)
{
    return auto_catch_tensor([t, scale]()
                             { return new torch::Tensor(*t * scale); },
                             err);
}

// Unscale and check for inf/nan (returns true if valid).
bool tensor_is_finite(tensor t)
{
    return torch::isfinite(*t).all().item<bool>();
}

// ============================================================================
// Vectorized Trading Environment Operations (CUDA)
// All operations work on batches of environments simultaneously.
// ============================================================================

// Vectorized position update: new_pos = action (0=flat, 1=long, 2=short).
// positions: [batch_size], actions: [batch_size]
tensor env_update_positions(char **err, tensor positions, tensor actions)
{
    return auto_catch_tensor([positions, actions]()
                             {
                                 // Map actions: 0->0 (flat), 1->1 (long), 2->-1 (short)
                                 auto new_pos = torch::where(
                                     *actions == 0,
                                     torch::zeros_like(*positions),
                                     torch::where(*actions == 1,
                                                  torch::ones_like(*positions),
                                                  -torch::ones_like(*positions)));
                                 return new torch::Tensor(new_pos);
                             },
                             err);
}

// Vectorized PnL calculation.
// positions: [batch_size], current_prices: [batch_size], entry_prices: [batch_size]
tensor env_calculate_pnl(char **err, tensor positions, tensor current_prices, tensor entry_prices)
{
    return auto_catch_tensor([positions, current_prices, entry_prices]()
                             {
                                 // PnL = position * (current_price - entry_price) / entry_price
                                 auto price_change = (*current_prices - *entry_prices) / (*entry_prices + 1e-8);
                                 return new torch::Tensor(*positions * price_change);
                             },
                             err);
}

// Vectorized fee calculation.
// volumes: [batch_size], fee_rate: scalar
tensor env_calculate_fees(char **err, tensor volumes, double fee_rate)
{
    return auto_catch_tensor([volumes, fee_rate]()
                             { return new torch::Tensor(*volumes * fee_rate); },
                             err);
}

// Vectorized reward calculation with risk-adjusted returns.
// pnl: [batch_size], fees: [batch_size], drawdown_penalty: scalar
tensor env_calculate_rewards(char **err, tensor pnl, tensor fees, double drawdown_penalty, tensor max_equity)
{
    return auto_catch_tensor([pnl, fees, drawdown_penalty, max_equity]()
                             {
                                 // Base reward = PnL - fees
                                 auto reward = *pnl - *fees;

                                 // Drawdown penalty (current equity vs max equity)
                                 auto equity = *pnl; // Simplified: equity = cumsum of pnl
                                 auto drawdown = (*max_equity - equity) / (*max_equity + 1e-8);
                                 auto penalty = drawdown * drawdown_penalty;

                                 return new torch::Tensor(reward - penalty);
                             },
                             err);
}

// Vectorized done check (max steps or bankruptcy).
// steps: [batch_size], max_steps: scalar, equity: [batch_size], min_equity: scalar
tensor env_check_done(char **err, tensor steps, int64_t max_steps, tensor equity, double min_equity)
{
    return auto_catch_tensor([steps, max_steps, equity, min_equity]()
                             {
                                 auto step_done = *steps >= max_steps;
                                 auto bankrupt = *equity < min_equity;
                                 return new torch::Tensor(step_done | bankrupt);
                             },
                             err);
}

// Vectorized state construction.
// Combines market features, position info, and account state into single tensor.
// market_features: [batch_size, feature_dim]
// positions: [batch_size]
// equity: [batch_size]
// Returns: [batch_size, feature_dim + 2]
tensor env_build_state(char **err, tensor market_features, tensor positions, tensor equity)
{
    return auto_catch_tensor([market_features, positions, equity]()
                             {
                                 auto pos_unsqueezed = positions->unsqueeze(-1);
                                 auto eq_unsqueezed = equity->unsqueeze(-1);
                                 return new torch::Tensor(torch::cat({*market_features, pos_unsqueezed, eq_unsqueezed}, -1));
                             },
                             err);
}

// Vectorized environment step (all-in-one).
// This is the main CUDA kernel that processes all environments in parallel.
// market_data: [batch_size, time_steps, feature_dim] - pre-loaded market data
// step_indices: [batch_size] - current time step for each env
// positions: [batch_size] - current position
// entry_prices: [batch_size] - entry price
// equity: [batch_size] - current equity
// max_equity: [batch_size] - max equity seen
// actions: [batch_size] - action to take
// fee_rate: scalar
// Returns struct with: next_state, rewards, dones, new_positions, new_equity, new_max_equity
void env_vectorized_step(char **err,
                         tensor market_data,
                         tensor step_indices,
                         tensor positions,
                         tensor entry_prices,
                         tensor equity,
                         tensor max_equity,
                         tensor actions,
                         double fee_rate,
                         double min_equity,
                         int64_t max_steps,
                         // Outputs (pre-allocated)
                         tensor out_states,
                         tensor out_rewards,
                         tensor out_dones,
                         tensor out_positions,
                         tensor out_entry_prices,
                         tensor out_equity,
                         tensor out_max_equity,
                         tensor out_step_indices)
{
    return auto_catch_void([=]()
                           {
        auto batch_size = positions->size(0);
        auto feature_dim = market_data->size(2);
        
        // Get current market features for each env.
        // market_data: [batch, time, features]
        // step_indices: [batch]
        auto indices = step_indices->unsqueeze(-1).unsqueeze(-1).expand({batch_size, 1, feature_dim});
        auto current_features = market_data->gather(1, indices).squeeze(1);  // [batch, features]
        auto current_prices = current_features.index({torch::indexing::Slice(), 3});  // Close price at index 3 (OHLCV)
        
        // Detect position changes.
        auto old_pos = *positions;
        auto new_pos = torch::where(
            *actions == 0,
            torch::zeros_like(old_pos),
            torch::where(*actions == 1,
                         torch::ones_like(old_pos),
                         -torch::ones_like(old_pos)));
        
        auto position_changed = old_pos != new_pos;
        
        // Calculate PnL for closing positions.
        auto close_pnl = torch::where(
            position_changed & (old_pos != 0),
            old_pos * (current_prices - *entry_prices) / (*entry_prices + 1e-8),
            torch::zeros_like(old_pos));
        
        // Calculate trading fees.
        auto fees = torch::where(
            position_changed,
            torch::abs(new_pos - old_pos) * current_prices * fee_rate,
            torch::zeros_like(old_pos));
        
        // Update equity.
        auto new_equity = *equity + close_pnl - fees;
        auto new_max_eq = torch::max(*max_equity, new_equity);
        
        // Update entry prices.
        auto new_entry = torch::where(
            position_changed & (new_pos != 0),
            current_prices,
            *entry_prices);
        
        // Calculate rewards.
        auto drawdown = (new_max_eq - new_equity) / (new_max_eq + 1e-8);
        auto rewards = close_pnl - fees - drawdown * 0.01;  // 1% drawdown penalty
        
        // Advance step.
        auto new_steps = *step_indices + 1;
        
        // Check done.
        auto dones = (new_steps >= max_steps) | (new_equity < min_equity);
        
        // Build next state.
        auto pos_unsqueezed = new_pos.unsqueeze(-1);
        auto eq_unsqueezed = (new_equity / 10000.0).unsqueeze(-1);  // Normalize equity
        auto next_state = torch::cat({current_features, pos_unsqueezed, eq_unsqueezed}, -1);
        
        // Copy to output tensors.
        out_states->copy_(next_state);
        out_rewards->copy_(rewards);
        out_dones->copy_(dones.to(torch::kFloat32));
        out_positions->copy_(new_pos);
        out_entry_prices->copy_(new_entry);
        out_equity->copy_(new_equity);
        out_max_equity->copy_(new_max_eq);
        out_step_indices->copy_(torch::where(dones, torch::zeros_like(new_steps), new_steps)); },
                           err);
}

// Reset environments that are done.
// dones: [batch_size] - which envs to reset
// initial_equity: scalar
void env_reset_done(char **err,
                    tensor dones,
                    tensor positions,
                    tensor entry_prices,
                    tensor equity,
                    tensor max_equity,
                    tensor step_indices,
                    double initial_equity)
{
    return auto_catch_void([=]()
                          {
        auto mask = dones->to(torch::kBool);
        positions->masked_fill_(mask, 0);
        entry_prices->masked_fill_(mask, 0);
        equity->masked_fill_(mask, initial_equity);
        max_equity->masked_fill_(mask, initial_equity);
        step_indices->masked_fill_(mask, 0); },
                          err);
}


// ============================================================================
// CUDA Utility Functions
// ============================================================================

bool is_cuda_available()
{
    return torch::cuda::is_available();
}

void cuda_synchronize()
{
    if (torch::cuda::is_available()) {
        torch::cuda::synchronize();
    }
}

void cuda_empty_cache()
{
    // Note: PyTorch C++ API doesn't expose empty_cache directly.
    // This is a no-op placeholder. Memory management happens via tensor destruction.
}

uint64_t cuda_memory_allocated()
{
    // Note: PyTorch C++ API doesn't expose memory_allocated directly.
    // Return 0 as placeholder.
    return 0;
}

uint64_t cuda_memory_total()
{
    if (!torch::cuda::is_available()) {
        return 0;
    }
    // Note: Getting total memory requires CUDA API directly.
    // Return 0 as placeholder - actual value would need cudaMemGetInfo.
    return 0;
}

const char* cuda_device_name()
{
    static char name[256] = "CUDA Device";
    if (torch::cuda::is_available()) {
        // Would need cudaGetDeviceProperties for actual name.
        return name;
    }
    return "No CUDA";
}

const char* cuda_sm_version()
{
    static char version[32] = "Unknown";
    return version;
}