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