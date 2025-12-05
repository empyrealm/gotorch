#ifndef __GOTORCH_TENSOR_H__
#define __GOTORCH_TENSOR_H__

#include "api.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // init
    GOTORCH_API tensor new_tensor(char **err);
    GOTORCH_API void free_tensor(tensor t);
    GOTORCH_API tensor tensor_arange(char **err, int end, int8_t dtype, int8_t device);
    GOTORCH_API tensor tensor_zeros(char **err, int64_t *shape, size_t shape_len, int8_t dtype, int8_t device);
    GOTORCH_API tensor tensor_from_data(char **err, void *data, int64_t *shape, size_t shape_len, int8_t dtype,
                                        int8_t device);
    GOTORCH_API void tensor_copy_data(tensor t, void *data);
    GOTORCH_API void tensor_set_data(char **err, tensor t, void *data);
    GOTORCH_API void tensor_set_requires_grad(char **err, tensor t, bool b);
    GOTORCH_API tensor tensor_to_device(char **err, tensor t, int8_t device);
    GOTORCH_API tensor tensor_to_scalar_type(char **err, tensor t, int8_t scalar_type);
    GOTORCH_API tensor tensor_clone(char **err, tensor t);
    // shapes
    GOTORCH_API tensor tensor_reshape(char **err, tensor t, int64_t *shape, size_t shape_len);
    GOTORCH_API tensor tensor_transpose(char **err, tensor t, int64_t dim1, int64_t dim2);
    GOTORCH_API tensor tensor_narrow(char **err, tensor t, int64_t dim, int64_t start, int64_t length);
    GOTORCH_API tensor tensor_vstack(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_hstack(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_view(char **err, tensor a, int64_t *shape, size_t shape_len);
    GOTORCH_API tensor tensor_permute(char **err, tensor a, int64_t *dims, size_t dims_len);
    // property
    GOTORCH_API size_t tensor_elem_size(tensor t);
    GOTORCH_API size_t tensor_elem_count(tensor t);
    GOTORCH_API int tensor_scalar_type(tensor t);
    GOTORCH_API size_t tensor_dims(tensor t);
    GOTORCH_API void tensor_shapes(tensor t, int64_t *shapes);
    GOTORCH_API int8_t tensor_device_type(tensor t);
    // utils
    GOTORCH_API tensor scaled_dot_product_attention(char **err, tensor q, tensor k, tensor v, tensor mask,
                                                    double dropout, bool is_causal);
    GOTORCH_API void clip_grad_norm(char **err, tensor *params, size_t params_count, double max_norm, double norm_type);
    GOTORCH_API void tensor_print(tensor t);
    GOTORCH_API tensor tensor_cat(char **err, tensor *tensors, size_t tensors_len, int64_t dim);
    GOTORCH_API tensor tensor_stack(char **err, tensor *tensors, size_t tensors_len, int64_t dim);
    GOTORCH_API tensor tensor_embedding(char **err, tensor weight, tensor indices, int64_t padding_idx);
    GOTORCH_API void svd(char **err, tensor t, tensor *u, tensor *s, tensor *v);
    GOTORCH_API tensor outer(char **err, tensor a, tensor b);
    GOTORCH_API tensor polar(char **err, tensor abs, tensor angle);
    GOTORCH_API tensor view_as_complex(char **err, tensor t);
    GOTORCH_API tensor view_as_real(char **err, tensor t);
    GOTORCH_API tensor tensor_flatten(char **err, tensor t, int64_t start_dim, int64_t end_dim);
    // init
    GOTORCH_API void init_kaiming_uniform(char **err, tensor t, double a);
    GOTORCH_API void init_xaiver_uniform(char **err, tensor t, double gain);
    GOTORCH_API void init_normal(char **err, tensor t, double mean, double std);
    GOTORCH_API void init_zeros(char **err, tensor t);
    GOTORCH_API tensor tensor_detach(tensor t);
    // serialization (.pt format)
    GOTORCH_API void tensors_save(char **err, tensor *tensors, size_t count, const char *path);
    GOTORCH_API size_t tensors_load(char **err, const char *path, tensor **out_tensors);
    GOTORCH_API void tensors_free_array(tensor *tensors, size_t count);
    // GPU-side sampling (avoid CPU transfer)
    GOTORCH_API tensor tensor_multinomial(char **err, tensor probs, int64_t num_samples, bool replacement);
    GOTORCH_API tensor tensor_categorical_sample(char **err, tensor logits);
    GOTORCH_API tensor tensor_normal_sample(char **err, tensor mean, tensor std);
    GOTORCH_API tensor tensor_argmax(char **err, tensor t, int64_t dim, bool keepdim);
    GOTORCH_API tensor tensor_rand(char **err, int64_t *shape, size_t shape_len, int8_t device);
    GOTORCH_API tensor tensor_randn(char **err, int64_t *shape, size_t shape_len, int8_t device);
    GOTORCH_API tensor tensor_clamp_minmax(char **err, tensor t, double min_val, double max_val);
    GOTORCH_API tensor tensor_where(char **err, tensor condition, tensor x, tensor y);
    // Indexing operations (for replay buffers)
    GOTORCH_API tensor tensor_index(char **err, tensor t, int64_t *indices, size_t indices_len);
    GOTORCH_API void tensor_index_put(char **err, tensor t, int64_t *indices, size_t indices_len, tensor value);
    GOTORCH_API void tensor_index_put_tensor(char **err, tensor t, tensor indices_tensor, tensor value);
    GOTORCH_API tensor tensor_index_select(char **err, tensor t, int64_t dim, tensor indices);
    // Global reductions
    GOTORCH_API tensor tensor_mean_all(char **err, tensor t);
    GOTORCH_API tensor tensor_sum_all(char **err, tensor t);
    GOTORCH_API tensor tensor_max_all(char **err, tensor t);
    GOTORCH_API tensor tensor_min_all(char **err, tensor t);
    GOTORCH_API tensor tensor_std_all(char **err, tensor t, bool unbiased);
    GOTORCH_API tensor tensor_pow_tensor(char **err, tensor t, tensor exp);
    GOTORCH_API tensor tensor_ones(char **err, int64_t *shape, size_t shape_len, int8_t dtype, int8_t device);
    // Mixed precision
    GOTORCH_API tensor tensor_half(char **err, tensor t);
    GOTORCH_API tensor tensor_bfloat16(char **err, tensor t);
    GOTORCH_API tensor tensor_float32(char **err, tensor t);
    GOTORCH_API bool tensor_is_half(tensor t);
    GOTORCH_API bool tensor_is_bfloat16(tensor t);
    GOTORCH_API tensor tensor_scale(char **err, tensor t, double scale);
    GOTORCH_API bool tensor_is_finite(tensor t);
    // Vectorized trading environment
    GOTORCH_API tensor env_update_positions(char **err, tensor positions, tensor actions);
    GOTORCH_API tensor env_calculate_pnl(char **err, tensor positions, tensor current_prices, tensor entry_prices);
    GOTORCH_API tensor env_calculate_fees(char **err, tensor volumes, double fee_rate);
    GOTORCH_API tensor env_calculate_rewards(char **err, tensor pnl, tensor fees, double drawdown_penalty, tensor max_equity);
    GOTORCH_API tensor env_check_done(char **err, tensor steps, int64_t max_steps, tensor equity, double min_equity);
    GOTORCH_API tensor env_build_state(char **err, tensor market_features, tensor positions, tensor equity);
    GOTORCH_API void env_vectorized_step(char **err,
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
                                         tensor out_states,
                                         tensor out_rewards,
                                         tensor out_dones,
                                         tensor out_positions,
                                         tensor out_entry_prices,
                                         tensor out_equity,
                                         tensor out_max_equity,
                                         tensor out_step_indices);
    GOTORCH_API void env_reset_done(char **err,
                                    tensor dones,
                                    tensor positions,
                                    tensor entry_prices,
                                    tensor equity,
                                    tensor max_equity,
                                    tensor step_indices,
                                    double initial_equity);

    // Comparison operations
    GOTORCH_API tensor tensor_eq(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_ne(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_lt(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_le(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_gt(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_ge(char **err, tensor a, tensor b);

    // Logical operations
    GOTORCH_API tensor tensor_logical_and(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_logical_or(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_logical_not(char **err, tensor t);

    // Element-wise min/max
    GOTORCH_API tensor tensor_maximum(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_minimum(char **err, tensor a, tensor b);

    // NaN handling
    GOTORCH_API tensor tensor_nan_to_num(char **err, tensor t, double nan_val, double posinf_val, double neginf_val);
    GOTORCH_API tensor tensor_isnan(char **err, tensor t);
    GOTORCH_API tensor tensor_isinf(char **err, tensor t);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_TENSOR_H__