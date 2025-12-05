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