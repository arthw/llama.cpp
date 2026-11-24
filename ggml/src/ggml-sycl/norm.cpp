#include "norm.hpp"

#include "fusion.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/presets.hpp"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <type_traits>

static void norm_debug_tensor(const char * function, const ggml_tensor * tensor) {
    if (!tensor) {
        return;
    }
    std::fprintf(stderr, "[norm] %s tensor=%s op=%s type=%s data=%p ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
                 function, tensor->name, ggml_op_name(tensor->op), ggml_type_name(tensor->type), tensor->data,
                 (long long) tensor->ne[0], (long long) tensor->ne[1],
                 (long long) tensor->ne[2], (long long) tensor->ne[3],
                 tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
    std::fprintf(stderr, "[norm] %s tensor=%s op_params_hex=", function, tensor->name);
    for (size_t index = 0; index < sizeof(tensor->op_params); ++index) {
        std::fprintf(stderr, "%02x", (unsigned) reinterpret_cast<const unsigned char *>(tensor->op_params)[index]);
    }
    std::fprintf(stderr, "\n");
}

template <typename Function>
static auto norm_debug_call(const char * function, ggml_backend_sycl_context & ctx,
                            std::initializer_list<const ggml_tensor *> tensors, Function operation,
                            ggml_tensor ** nodes = nullptr, int count = 0) -> decltype(operation()) {
    const char * debug_env = std::getenv("GGML_SYCL_DEBUG_RMS_NORM");
    if (!debug_env || debug_env[0] == '\0' || std::strcmp(debug_env, "0") == 0) {
        return operation();
    }
    const char * stage = "entry";
    try {
        std::fprintf(stderr, "[norm] %s ENTER device=%d subgroup=%d configured_max_wg=%d batch_count=%d\n",
                     function, ctx.device, WARP_SIZE, ggml_sycl_info().max_work_group_sizes[ctx.device], count);
        std::fflush(stderr);
        const auto log_tensor = [&](const ggml_tensor * tensor) {
            norm_debug_tensor(function, tensor);
            if (tensor) {
                for (const ggml_tensor * source : tensor->src) {
                    norm_debug_tensor(function, source);
                }
            }
        };
        for (const ggml_tensor * tensor : tensors) {
            log_tensor(tensor);
        }
        for (int index = 0; nodes && index < count; ++index) {
            std::fprintf(stderr, "[norm] %s batch_index=%d\n", function, index);
            log_tensor(nodes[index]);
        }
        stage = "pre-operation wait";
        std::fprintf(stderr, "[norm] %s %s\n", function, stage);
        std::fflush(stderr);
        ctx.stream()->wait_and_throw();
        stage = "operation";
        std::fprintf(stderr, "[norm] %s %s\n", function, stage);
        std::fflush(stderr);
        const auto finish = [&]() {
            stage = "post-operation wait";
            std::fprintf(stderr, "[norm] %s %s\n", function, stage);
            std::fflush(stderr);
            ctx.stream()->wait_and_throw();
            std::fprintf(stderr, "[norm] %s COMPLETED\n", function);
            std::fflush(stderr);
        };
        if constexpr (std::is_void_v<decltype(operation())>) {
            operation();
            finish();
        } else {
            const auto result = operation();
            std::fprintf(stderr, "[norm] %s fusion_accepted=%d\n", function, int(result));
            finish();
            return result;
        }
    } catch (const sycl::exception & exception) {
        std::fprintf(stderr, "[norm] %s FAILED stage=%s device=%d code=%d category=%s what=%s\n",
                     function, stage, ctx.device, exception.code().value(),
                     exception.code().category().name(), exception.what());
        std::fflush(stderr);
        throw;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "[norm] %s FAILED stage=%s device=%d what=%s\n",
                     function, stage, ctx.device, exception.what());
        std::fflush(stderr);
        throw;
    }
}

static void norm_f32(const float* x, float* dst, const int ncols,
    const int64_t src_stride_col, const int64_t src_stride_row, const int64_t src_stride_channel, const int64_t src_stride_sample,
    const int64_t dst_stride_col, const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
    const float eps, const sycl::nd_item<3>& item_ct1, sycl::float2* s_sum, int block_size) {

    const int nthreads = item_ct1.get_local_range(2);
    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int tid = item_ct1.get_local_id(2);
    const int nwarps = nthreads / WARP_SIZE;

    const auto src_offset = calculate_offset<3>({src_stride_sample, src_stride_channel, src_stride_row}, {sample, channel, row});
    const auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x += src_offset;
    dst += dst_offset;

    sycl::float2 mean_var = sycl::float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col * src_stride_col];
        mean_var.x() += xi;
        mean_var.y() += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var, item_ct1);
    if  (block_size > WARP_SIZE) {
        const auto sub_group = item_ct1.get_sub_group();
        const auto sg_id = sub_group.get_group_linear_id();
        const auto wi_in_sg = sub_group.get_local_linear_id();
        if (wi_in_sg == 0) {
            s_sum[sg_id] = mean_var;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        mean_var = 0.f;
        const size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        for (size_t i = 0; i < nreduce; i += 1)
        {
            mean_var += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        mean_var = warp_reduce_sum(mean_var, item_ct1);
    }

    const float mean = mean_var.x() / ncols;
    const float var = mean_var.y() / ncols - mean * mean;
    const float inv_std = sycl::rsqrt(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col * dst_stride_col] = (x[col * src_stride_col] - mean) * inv_std;
    }
}

static void group_norm_f32(const float* x, float* dst, const int group_size, const int ne_elements, const float eps,
    const sycl::nd_item<3>& item_ct1, float* s_sum, int block_size) {
    int start = item_ct1.get_group(2) * group_size;
    int end = start + group_size;
    const int nthreads = item_ct1.get_local_range(2);
    const int nwarps = nthreads / WARP_SIZE;
    start += item_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_SIZE;

    if (end >= ne_elements) {
        end = ne_elements;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {

        int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        /*
        DPCT1118:1: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:54: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {

        int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        /*
        DPCT1118:2: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:55: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    float variance = tmp / group_size;
    float scale = sycl::rsqrt(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <bool do_multiply = false, bool do_add = false>
static void rms_norm_f32(const float* x, float* dst, const int ncols,
    const int64_t src_stride_col, const int64_t src_stride_row, const int64_t src_stride_channel, const int64_t src_stride_sample,
    const int64_t dst_stride_col, const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
    const float eps, const sycl::nd_item<3>& item_ct1, float* s_sum, int block_size,
    const float* mul = nullptr, const int64_t mul_stride_row = 0, const int64_t mul_stride_channel = 0,
    const int64_t mul_stride_sample = 0, const int mul_nrows = 0, const int mul_nchannels = 0, const int mul_nsamples = 0,
    const float* add = nullptr, const int64_t add_stride_row = 0, const int64_t add_stride_channel = 0,
    const int64_t add_stride_sample = 0, const int add_nrows = 0, const int add_nchannels = 0, const int add_nsamples = 0,
    const float scale_mul = 1.0f) {

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int nthreads = item_ct1.get_local_range(2);

    const int tid = item_ct1.get_local_id(2);
    const int nwarps = nthreads / WARP_SIZE;

    const auto src_offset = calculate_offset<3>({src_stride_sample, src_stride_channel, src_stride_row}, {sample, channel, row});
    const auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x   += src_offset;
    dst += dst_offset;

    if constexpr (do_multiply) {
        const int mul_row     = row     % mul_nrows;
        const int mul_channel = channel % mul_nchannels;
        const int mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = row     % add_nrows;
        const int add_channel = channel % add_nchannels;
        const int add_sample  = sample  % add_nsamples;
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    // float4 path for contiguous, 16-byte aligned rows (decode), views and odd lengths stay strided
    bool contig = src_stride_col == 1 && dst_stride_col == 1 && ncols % 4 == 0 && (uintptr_t) x % 16 == 0 &&
                  (uintptr_t) dst % 16 == 0;
    if constexpr (do_multiply) {
        contig = contig && (uintptr_t) mul % 16 == 0;
    }
    if constexpr (do_add) {
        contig = contig && (uintptr_t) add % 16 == 0;
    }
    const int n4 = ncols / 4;

    if (contig) {
        const sycl::float4 * x4 = reinterpret_cast<const sycl::float4 *>(x);
        for (int i = tid; i < n4; i += block_size) {
            const sycl::float4 v = x4[i];
            tmp += sycl::dot(v, v);
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            const float xi = x[col * src_stride_col];
            tmp += xi * xi;
        }
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {
        const auto sub_group = item_ct1.get_sub_group();
        const auto sg_id = sub_group.get_group_linear_id();
        const auto wi_in_sg = sub_group.get_local_linear_id();
        if (wi_in_sg == 0) {
            s_sum[sg_id] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        const size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    const float mean = tmp / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    if (contig) {
        const sycl::float4 * x4 = reinterpret_cast<const sycl::float4 *>(x);
        sycl::float4 *       d4 = reinterpret_cast<sycl::float4 *>(dst);
        const sycl::float4 * m4 = reinterpret_cast<const sycl::float4 *>(mul);
        const sycl::float4 * a4 = reinterpret_cast<const sycl::float4 *>(add);
        for (int i = tid; i < n4; i += block_size) {
            const sycl::float4 v = x4[i];
            if constexpr (do_multiply && do_add) {
                const sycl::float4 m = m4[i];
                const sycl::float4 a = a4[i];
                d4[i]                = scale * v * m + a;
            } else if constexpr (do_multiply) {
                const sycl::float4 m = m4[i];
                d4[i]                = scale * v * m;
            } else {
                d4[i] = scale * v * scale_mul;
            }
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            if constexpr (do_multiply && do_add) {
                dst[col * dst_stride_col] = scale * x[col * src_stride_col] * mul[col] + add[col];
            } else if constexpr (do_multiply) {
                dst[col * dst_stride_col] = scale * x[col * src_stride_col] * mul[col];
            } else {
                // folded epilogue of a fused GGML_OP_SCALE consumer (qwen35 GDN l2 norms);
                // the explicit temporary keeps the float evaluation order identical to
                // running rms_norm and scale as two separate kernels
                const float v             = scale * x[col * src_stride_col];
                dst[col * dst_stride_col] = v * scale_mul;
            }
        }
    }
}

template<int warp_size>
static void l2_norm_f32(const float * x, float * dst, const int ncols,
    const int64_t src_stride_col, const int64_t src_stride_row, const int64_t src_stride_channel,
    const int64_t src_stride_sample, const int64_t dst_stride_col, const int64_t dst_stride_row,
    const int64_t dst_stride_channel, const int64_t dst_stride_sample, const float eps,
    const sycl::nd_item<3>& item_ct1, float* s_sum, const int block_size) {

    const int row     = item_ct1.get_group(2);
    const int channel = item_ct1.get_group(1);
    const int sample  = item_ct1.get_group(0);
    const int tid     = item_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + row*src_stride_row;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + row*dst_stride_row;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col * src_stride_col];
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, warp_size>(tmp, s_sum, block_size);
    const float scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col * dst_stride_col] = scale * x[col * src_stride_col];
    }
}

static void norm_f32_sycl(const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
    const int64_t src_stride_col, const int64_t src_stride_row, const int64_t src_stride_channel, const int64_t src_stride_sample,
    const int64_t dst_stride_col, const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
        const float eps, queue_ptr stream, int device) {

    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    printf("Launching norm_f32_sycl with global_dims: (%d, %d, %d)\n", nsamples, nchannels, nrows);
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        printf("Launching norm_f32_sycl with block_dims: (1, 1, %d)\n", WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE);
                });
            });
    }
    else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        printf("Launching norm_f32_sycl with block_dims: (1, 1, %d)\n", work_group_size);
        printf("work_group_size / WARP_SIZE: %d\n", work_group_size / WARP_SIZE);
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<sycl::float2, 1> s_sum_acc_ct1(
                            sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

static void group_norm_f32_sycl(const float* x, float* dst,
    const int num_groups, const float eps, const int group_size,
    const int ne_elements, queue_ptr stream, int device) {
    if (group_size < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            const float eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    group_norm_f32(
                        x, dst, group_size, ne_elements, eps_ct4, item_ct1,
                        nullptr, WARP_SIZE);
                });
            });
    }
    else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);

            const float eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    group_norm_f32(x, dst, group_size, ne_elements,
                        eps_ct4, item_ct1,
                        get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

static void rms_norm_f32_sycl(const float* x, float* dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
    const int64_t src_stride_col, const int64_t src_stride_row, const int64_t src_stride_channel, const int64_t src_stride_sample,
    const int64_t dst_stride_col, const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
    const float eps, queue_ptr stream, int device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    printf("Launching rms_norm_f32_sycl with global_dims: (%d, %d, %d)\n", nsamples, nchannels, nrows);
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        printf("Launching rms_norm_f32_sycl with block_dims: (1, 1, %d)\n", WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE);
                });
            });
    }
    else {

        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        printf("Launching rms_norm_f32_sycl with work_group_size: %d\n", work_group_size);
        printf("work_group_size / WARP_SIZE: %d\n", work_group_size / WARP_SIZE);
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

static void rms_norm_scale_f32_sycl(const float* x, float* dst, const int ncols, const int nrows,
    const int nchannels, const int nsamples,
    const int64_t src_stride_col, const int64_t src_stride_row, const int64_t src_stride_channel, const int64_t src_stride_sample,
    const int64_t dst_stride_col, const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
    const float eps, const float scale_mul, queue_ptr stream, int device) {
    printf("debug rms_norm_scale_f32_sycl\n");
    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    printf("debug global_dims: nsamples=%d, nchannels=%d, nrows=%d\n", nsamples, nchannels, nrows);
    printf("debug ncols=%d\n", ncols);
    if (ncols < 1024) {
        printf("debug using small ncols path\n");
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE,
                        nullptr, 0, 0, 0, 0, 0, 0,
                        nullptr, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
    else {
        printf("debug using large ncols path\n");
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        printf("debug work_group_size=%d\n", work_group_size);
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        printf("debug local accessor size=%d\n", work_group_size / WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size,
                        nullptr, 0, 0, 0, 0, 0, 0,
                        nullptr, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
    printf("debug rms_norm_scale_f32_sycl done\n");
}

static void rms_norm_mul_f32_sycl(const float* x, const float* mul, float* dst, const int ncols, const int nrows,
        const int nchannels, const int nsamples,
        const int64_t src_stride_col, const int64_t src_stride_row, const int64_t src_stride_channel, const int64_t src_stride_sample,
        const int64_t dst_stride_col, const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const int mul_nrows, const int mul_nchannels, const int mul_nsamples,
        const float eps, queue_ptr stream, int device) {
    const char * debug_env = std::getenv("GGML_SYCL_DEBUG_RMS_NORM");
    const bool debug = debug_env && debug_env[0] != '\0' && std::strcmp(debug_env, "0") != 0;
    const char * stage = "device info";
    try {
        if (debug) {
            const auto   sycl_device     = stream->get_device();
            const int    work_group_size = ncols < 1024 ? WARP_SIZE : ggml_sycl_info().max_work_group_sizes[device];
            const size_t local_elements  = ncols < 1024 ? 0 : work_group_size / WARP_SIZE;
            const auto   max_items       = sycl_device.get_info<sycl::info::device::max_work_item_sizes<3>>();
            std::fprintf(stderr, "[rms_norm_mul] device=%d name=%s driver=%s shape=[%d,%d,%d,%d] eps=%g\n", device,
                         sycl_device.get_info<sycl::info::device::name>().c_str(),
                         sycl_device.get_info<sycl::info::device::driver_version>().c_str(), ncols, nrows, nchannels,
                         nsamples, eps);
            std::fprintf(
                stderr,
                "[rms_norm_mul] global=[%zu,%zu,%zu] local=[1,1,%d] subgroup=%d s_sum_elements=%zu s_sum_bytes=%zu\n",
                size_t(nsamples), size_t(nchannels), size_t(nrows) * work_group_size, work_group_size, WARP_SIZE,
                local_elements, local_elements * sizeof(float));
            std::fprintf(
                stderr,
                "[rms_norm_mul] device_max_wg=%zu max_items=[%zu,%zu,%zu] local_mem_bytes=%llu supported_subgroups=",
                sycl_device.get_info<sycl::info::device::max_work_group_size>(), max_items[0], max_items[1],
                max_items[2], (unsigned long long) sycl_device.get_info<sycl::info::device::local_mem_size>());
            for (const auto subgroup_size : sycl_device.get_info<sycl::info::device::sub_group_sizes>()) {
                std::fprintf(stderr, " %zu", subgroup_size);
            }
            std::fprintf(stderr, "\n[rms_norm_mul] x=%p mul=%p dst=%p alignment_mod16=[%zu,%zu,%zu]\n",
                         (const void *) x, (const void *) mul, (void *) dst, size_t((uintptr_t) x % 16),
                         size_t((uintptr_t) mul % 16), size_t((uintptr_t) dst % 16));
            stage = "pre-launch wait";
            std::fprintf(stderr, "[rms_norm_mul] %s\n", stage);
            std::fflush(stderr);
            stream->wait_and_throw();
        }
        stage = "submit";
        if (debug) {
            std::fprintf(stderr, "[rms_norm_mul] %s path=%s\n", stage,
                         ncols < 1024 ? "single-subgroup" : "multi-subgroup");
            std::fflush(stderr);
        }
        const sycl::range<3> global_dims(nsamples, nchannels, nrows);
        if (debug) {
            std::fprintf(stderr, "[rms_norm_mul] global_dims=[%d,%d,%d]\n", nsamples, nchannels, nrows);
            std::fflush(stderr);
        }
        if (ncols < 1024) {
            const sycl::range<3> block_dims(1, 1, WARP_SIZE);
            stream->submit([&](sycl::handler & cgh) {
                cgh.parallel_for(sycl::nd_range<3>(global_dims * block_dims, block_dims),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                     rms_norm_f32<true>(
                                         x, dst, ncols, src_stride_col, src_stride_row, src_stride_channel,
                                         src_stride_sample, dst_stride_col, dst_stride_row, dst_stride_channel,
                                         dst_stride_sample, eps, item_ct1, nullptr, WARP_SIZE, mul, mul_stride_row,
                                         mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                                 });
            });
        } else {
            const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
            assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
            const sycl::range<3> block_dims(1, 1, work_group_size);
            if (debug) {
                std::fprintf(stderr, "[rms_norm_mul] work_group_size=%d\n", work_group_size);
                std::fprintf(stderr, "[rms_norm_mul] work_group_size / WARP_SIZE=%d\n", work_group_size / WARP_SIZE);
                std::fflush(stderr);
            }
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
                cgh.parallel_for(sycl::nd_range<3>(global_dims * block_dims, block_dims),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                     rms_norm_f32<true>(x, dst, ncols, src_stride_col, src_stride_row,
                                                        src_stride_channel, src_stride_sample, dst_stride_col,
                                                        dst_stride_row, dst_stride_channel, dst_stride_sample, eps,
                                                        item_ct1, get_pointer(s_sum_acc_ct1), work_group_size, mul,
                                                        mul_stride_row, mul_stride_channel, mul_stride_sample,
                                                        mul_nrows, mul_nchannels, mul_nsamples);
                                 });
            });
        }
        if (debug) {
            stage = "post-launch wait";
            std::fprintf(stderr, "[rms_norm_mul] %s\n", stage);
            std::fflush(stderr);
            stream->wait_and_throw();
            std::fprintf(stderr, "[rms_norm_mul] completed\n");
            std::fflush(stderr);
        }
    } catch (const sycl::exception & exception) {
        std::fprintf(stderr,
                     "[rms_norm_mul] FAILED stage=%s device=%d shape=[%d,%d,%d,%d] code=%d category=%s what=%s\n",
                     stage, device, ncols, nrows, nchannels, nsamples, exception.code().value(),
                     exception.code().category().name(), exception.what());
        std::fflush(stderr);
        throw;
    }
}

static void rms_norm_mul_add_f32_sycl(const float * x,
                                      const float * mul,
                                      const float * add,
                                      float *       dst,
                                      const int     ncols,
                                      const int     nrows,
                                      const int     nchannels,
                                      const int     nsamples,
                                      const int64_t src_stride_col,
                                      const int64_t src_stride_row,
                                      const int64_t src_stride_channel,
                                      const int64_t src_stride_sample,
                                      const int64_t dst_stride_col,
                                      const int64_t dst_stride_row,
                                      const int64_t dst_stride_channel,
                                      const int64_t dst_stride_sample,
                                      const int64_t mul_stride_row,
                                      const int64_t mul_stride_channel,
                                      const int64_t mul_stride_sample,
                                      const int     mul_nrows,
                                      const int     mul_nchannels,
                                      const int     mul_nsamples,
                                      const int64_t add_stride_row,
                                      const int64_t add_stride_channel,
                                      const int64_t add_stride_sample,
                                      const int     add_nrows,
                                      const int     add_nchannels,
                                      const int     add_nsamples,
                                      const float   eps,
                                      queue_ptr     stream,
                                      int           device) {
    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<true, true>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        add, add_stride_row, add_stride_channel, add_stride_sample, add_nrows, add_nchannels, add_nsamples);
                });
        });
    } else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<true, true>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        add, add_stride_row, add_stride_channel, add_stride_sample, add_nrows, add_nchannels, add_nsamples);
                });
            });
    }
}

template<int warp_size>
static void l2_norm_f32_sycl(const float *   x,
                             float *         dst,
                             const int       ncols,
                             const int       nrows,
                             const int       nchannels,
                             const int       nsamples,
                             const int64_t   src_stride_col,
                             const int64_t   src_stride_row,
                             const int64_t   src_stride_channel,
                             const int64_t   src_stride_sample,
                             const int64_t   dst_stride_col,
                             const int64_t   dst_stride_row,
                             const int64_t   dst_stride_channel,
                             const int64_t   dst_stride_sample,
                             const float     eps,
                             queue_ptr       stream,
                             int             device) {
    const dpct::dim3 blocks_num(nrows, nchannels, nsamples);

    if (ncols < 1024) {
        const dpct::dim3 block_dims(warp_size, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(blocks_num * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(warp_size)]] {
                    l2_norm_f32<warp_size>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1,
                        nullptr, warp_size);
                });
            });
    }
    else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (warp_size * warp_size) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        int lsm_size =  block_dims[2] > warp_size ? work_group_size / warp_size * sizeof(float): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(lsm_size),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(blocks_num * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(warp_size)]] {
                    l2_norm_f32<warp_size>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

// Batched L2 norm: N independent same-shape F32 tensors in one launch; the tensor
// index is folded into grid dim0 and each row's reduction is identical to the
// single-tensor kernel, so the result is bit-exact.
struct l2_batch_ptrs {
    const float * src[GGML_SYCL_L2_BATCH_MAX];
    float *       dst[GGML_SYCL_L2_BATCH_MAX];
};

// One stride set shared by the whole batch: the caller only groups tensors whose nb[]
// all match, so per-tensor state stays two pointers.
struct l2_batch_strides {
    int     ne1, ne2;
    int64_t ss0, ss1, ss2, ss3;
    int64_t ds0, ds1, ds2, ds3;
};

template <int warp_size>
static void l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, const int ncols, const float eps,
                              const sycl::nd_item<3> & item_ct1) {
    const int t   = item_ct1.get_group(0);  // tensor index
    const int r   = item_ct1.get_group(2);  // flattened row over ne1*ne2*ne3
    const int tid = item_ct1.get_local_id(2);

    const int i1 = r % st.ne1;
    const int i2 = (r / st.ne1) % st.ne2;
    const int i3 = r / (st.ne1 * st.ne2);

    const float * x   = p.src[t] + i3 * st.ss3 + i2 * st.ss2 + i1 * st.ss1;
    float *       dst = p.dst[t] + i3 * st.ds3 + i2 * st.ds2 + i1 * st.ds1;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += warp_size) {
        const float xi = x[col * st.ss0];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, warp_size>(tmp, (float *) nullptr, warp_size);
    const float scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    for (int col = tid; col < ncols; col += warp_size) {
        dst[col * st.ds0] = scale * x[col * st.ss0];
    }
}

template <int warp_size>
static void l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, const int n_tensors,
                                   const int ncols, const int nrows_total, const float eps,
                                   queue_ptr stream) {
    const dpct::dim3 blocks_num(nrows_total, 1, n_tensors);
    const dpct::dim3 block_dims(warp_size, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(blocks_num * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                l2_norm_f32_batch<warp_size>(p, st, ncols, eps, item_ct1);
            });
    });
}

void ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    return norm_debug_call(__func__, ctx, { dst }, [&]() {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);
    const size_t ts0 = ggml_type_size(src0->type);
    const size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    const int64_t ss0 = nb00 / ts0;
    const int64_t ss1 = nb01 / ts0;
    const int64_t ss2 = nb02 / ts0;
    const int64_t ss3 = nb03 / ts0;
    const int64_t ds0 = nb0 / tdst;
    const int64_t ds1 = nb1 / tdst;
    const int64_t ds2 = nb2 / tdst;
    const int64_t ds3 = nb3 / tdst;

    norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_stream, ctx.device);
    });
}

void ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    return norm_debug_call(__func__, ctx, { dst }, [&]() {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));

    int group_size = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_sycl(src0_dd, dst_dd, num_groups, eps, group_size, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_stream, ctx.device);
    });
}

void ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    return norm_debug_call(__func__, ctx, { dst }, [&]() {

    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS
    const size_t ts0 = ggml_type_size(src0->type);
    const size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    const int64_t ss0 = nb00 / ts0;
    const int64_t ss1 = nb01 / ts0;
    const int64_t ss2 = nb02 / ts0;
    const int64_t ss3 = nb03 / ts0;
    const int64_t ds0 = nb0 / tdst;
    const int64_t ds1 = nb1 / tdst;
    const int64_t ds2 = nb2 / tdst;
    const int64_t ds3 = nb3 / tdst;
    rms_norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_stream, ctx.device);
    });
}

// Fused rms_norm + scale (the qwen35 GDN l2-norm pair build_gdn_l2_norm emits):
// the scale factor is a host scalar in the GGML_OP_SCALE node's op_params, so
// unlike the mul variants there is no second device tensor to wire up.
void ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                       ggml_tensor * scale_tensor) {
    return norm_debug_call(__func__, ctx, { dst, scale_tensor }, [&]() {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(src0->data);
    float *       dst_dd  = static_cast<float *>(scale_tensor->data);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    float scale_mul;
    memcpy(&scale_mul, scale_tensor->op_params, sizeof(float));
    GGML_ASSERT(scale_mul >= 0.0f);

    GGML_TENSOR_UNARY_OP_LOCALS
    const size_t ts0 = ggml_type_size(src0->type);
    const size_t tdst = ggml_type_size(scale_tensor->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(scale_tensor->nb[0] % tdst == 0 && scale_tensor->nb[1] % tdst == 0 &&
                scale_tensor->nb[2] % tdst == 0 && scale_tensor->nb[3] % tdst == 0);
    const int64_t ss0 = nb00 / ts0;
    const int64_t ss1 = nb01 / ts0;
    const int64_t ss2 = nb02 / ts0;
    const int64_t ss3 = nb03 / ts0;
    const int64_t ds0 = scale_tensor->nb[0] / tdst;
    const int64_t ds1 = scale_tensor->nb[1] / tdst;
    const int64_t ds2 = scale_tensor->nb[2] / tdst;
    const int64_t ds3 = scale_tensor->nb[3] / tdst;
    rms_norm_scale_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, scale_mul, main_stream, ctx.device);
    });
}

void ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    return norm_debug_call(__func__, ctx, { dst, mul_tensor }, [&]() {
    const ggml_tensor * rms_norm_src = dst->src[0];
    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));

    const float *       src0_dd = static_cast<const float *>(rms_norm_src->data);
    const float *       mul_dd  = nullptr;
    const ggml_tensor * mul_src = nullptr;
    if (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<const float *>(mul_tensor->src[1]->data);
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<const float *>(mul_tensor->src[0]->data);
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }
    float * dst_dd = static_cast<float *>(mul_tensor->data);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s00 = rms_norm_src->nb[0] / ts0;
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t tdst = ggml_type_size(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    const int64_t d00 = mul_tensor->nb[0] / tdst;
    const int64_t d01 = mul_tensor->nb[1] / tdst;
    const int64_t d02 = mul_tensor->nb[2] / tdst;
    const int64_t d03 = mul_tensor->nb[3] / tdst;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const char * debug_env = std::getenv("GGML_SYCL_DEBUG_RMS_NORM");
    if (debug_env && debug_env[0] != '\0' && std::strcmp(debug_env, "0") != 0) {
        const ggml_tensor * tensors[] = { rms_norm_src, dst, mul_src, mul_tensor };
        for (const ggml_tensor * tensor : tensors) {
            std::fprintf(stderr, "[rms_norm_mul] tensor=%s op=%s type=%s data=%p ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
                         tensor->name, ggml_op_name(tensor->op), ggml_type_name(tensor->type), tensor->data,
                         (long long) tensor->ne[0], (long long) tensor->ne[1],
                         (long long) tensor->ne[2], (long long) tensor->ne[3],
                         tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
        }
        std::fflush(stderr);
    }

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_stream, ctx.device);
    });
}

void ggml_sycl_op_rms_norm_fused_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * mul_tensor, ggml_tensor * add_tensor) {
    return norm_debug_call(__func__, ctx, { dst, mul_tensor, add_tensor }, [&]() {
    const ggml_tensor * rms_norm_src = dst->src[0];
    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));

    const float *       src0_dd = static_cast<const float *>(rms_norm_src->data);
    const float *       mul_dd  = nullptr;
    const ggml_tensor * mul_src = nullptr;
    if (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<const float *>(mul_tensor->src[1]->data);
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<const float *>(mul_tensor->src[0]->data);
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float *       add_dd  = nullptr;
    const ggml_tensor * add_src = nullptr;
    if (add_tensor->src[0] == mul_tensor) {
        add_dd  = static_cast<const float *>(add_tensor->src[1]->data);
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_dd  = static_cast<const float *>(add_tensor->src[0]->data);
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_dd = static_cast<float *>(add_tensor->data);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s00 = rms_norm_src->nb[0] / ts0;
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t tdst = ggml_type_size(add_tensor->type);
    GGML_ASSERT(add_tensor->nb[0] == tdst);
    const int64_t d00 = add_tensor->nb[0] / tdst;
    const int64_t d01 = add_tensor->nb[1] / tdst;
    const int64_t d02 = add_tensor->nb[2] / tdst;
    const int64_t d03 = add_tensor->nb[3] / tdst;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_add_f32_sycl(src0_dd, mul_dd, add_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples,
        add_s01, add_s02, add_s03, add_nrows, add_nchannels, add_nsamples, eps, main_stream, ctx.device);
    });
}

// ADD(a, b) -> RMS_NORM -> MUL(w) in one launch. The sum (read by a later residual add) and
// normed * w are written after a barrier, as mul->data may alias a.
static void add_rms_norm_mul_f32(const float *            a,
                                 const float *            b,
                                 float *                  sum,
                                 const float *            mul,
                                 float *                  dst,
                                 const int                ncols,
                                 const float              eps,
                                 const sycl::nd_item<3> & item_ct1,
                                 float *                  s_sum,
                                 const int                block_size) {
    const int    row = item_ct1.get_group(2);
    const size_t off = (size_t) row * ncols;
    const int    tid = item_ct1.get_local_id(2);
    const int    n4  = ncols / 4;

    const sycl::float4 * a4 = reinterpret_cast<const sycl::float4 *>(a + off);
    const sycl::float4 * b4 = reinterpret_cast<const sycl::float4 *>(b + off);

    sycl::float4 v[2];
    float        tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int i = tid + k * block_size;
        if (i < n4) {
            const sycl::float4 va = a4[i];
            const sycl::float4 vb = b4[i];
            v[k]                  = va + vb;
            tmp += sycl::dot(v[k], v[k]);
        }
    }

    const int nwarps = block_size / WARP_SIZE;
    tmp              = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {
        const auto sub_group = item_ct1.get_sub_group();
        if (sub_group.get_local_linear_id() == 0) {
            s_sum[sub_group.get_group_linear_id()] = tmp;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        const size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        tmp                  = 0.0f;
        for (size_t i = 0; i < nreduce; i += 1) {
            tmp += s_sum[sub_group.get_local_linear_id() + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    const float mean  = tmp / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    sycl::float4 *       s4 = reinterpret_cast<sycl::float4 *>(sum + off);
    sycl::float4 *       d4 = reinterpret_cast<sycl::float4 *>(dst + off);
    const sycl::float4 * m4 = reinterpret_cast<const sycl::float4 *>(mul);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int i = tid + k * block_size;
        if (i < n4) {
            const sycl::float4 m = m4[i];
            const sycl::float4 u = scale * v[k] * m;
            s4[i]                = v[k];
            d4[i]                = u;
        }
    }
}

static void add_rms_norm_mul_f32_sycl(const float * a,
                                      const float * b,
                                      float *       sum,
                                      const float * mul,
                                      float *       dst,
                                      const int     ncols,
                                      const int     nrows,
                                      const float   eps,
                                      queue_ptr     stream,
                                      int           device) {
    const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
    GGML_ASSERT(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
    const sycl::range<3> global_dims(1, 1, nrows);
    const sycl::range<3> block_dims(1, 1, work_group_size);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> s_sum(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
        cgh.parallel_for(sycl::nd_range<3>(global_dims * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             add_rms_norm_mul_f32(a, b, sum, mul, dst, ncols, eps, item_ct1, get_pointer(s_sum),
                                                  work_group_size);
                         });
    });
}

bool ggml_sycl_op_add_rms_norm_fused(ggml_backend_sycl_context & ctx,
                                     ggml_tensor *               add,
                                     ggml_tensor *               rms,
                                     ggml_tensor *               mul) {
    return norm_debug_call(__func__, ctx, { add, rms, mul }, [&]() {
    ggml_tensor * a     = add->src[0];
    ggml_tensor * b     = add->src[1];
    ggml_tensor * w     = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const int64_t ncols = add->ne[0];
    const int     wg    = ggml_sycl_info().max_work_group_sizes[ctx.device];
    // whole rows of float4, at most two per work-item, from 16-byte aligned contiguous tensors
    if (add->ne[2] != 1 || add->ne[3] != 1 || ncols < 1024 || ncols % 4 != 0 || ncols > 8 * (int64_t) wg ||
        w->ne[1] != 1) {
        return false;
    }
    for (const ggml_tensor * t : { a, b, add, mul, w }) {
        if (!ggml_is_contiguous(t) || (uintptr_t) t->data % 16 != 0) {
            return false;
        }
    }
    float eps = 0.0f;
    memcpy(&eps, rms->op_params, sizeof(float));
    if (!(eps >= 0.0f)) {
        return false;
    }
    add_rms_norm_mul_f32_sycl((const float *) a->data, (const float *) b->data, (float *) add->data,
                              (const float *) w->data, (float *) mul->data, (int) ncols, (int) add->ne[1], eps,
                              ctx.stream(), ctx.device);
    return true;
    });
}

void ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    return norm_debug_call(__func__, ctx, { dst }, [&]() {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    float eps = 1e-5f;
    std::memcpy(&eps, dst->op_params, sizeof(float));
    if (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    const float * g_base  = static_cast<const float *>(dst->src[0]->data); // dz
    const float * x_base  = static_cast<const float *>(dst->src[1]->data); // x
          float * dx_base = static_cast<      float *>(dst->data);

    const int64_t D  = dst->ne[0];
    const int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (void) n3;
    const int64_t N  = ggml_nrows(dst);
    if (D == 0 || N == 0) return;

    const ggml_tensor *G = dst->src[0];
    const ggml_tensor *X = dst->src[1];
    const int ts = (int) ggml_type_size(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    const int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    const int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    const int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::queue_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    const int device_max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
    auto roundup = [](int v, int m) { return ((v + m - 1) / m) * m; };
    int wg_cap = 256;
    if (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    int WG = std::max(WARP_SIZE, std::min(roundup((int)std::min<int64_t>(D, wg_cap), WARP_SIZE), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        const int nwarps_loc = std::max(1, WG / WARP_SIZE);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        auto l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        auto l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int row = item_ct1.get_group(2);
                const int tid = item_ct1.get_local_id(2);

                const int64_t i1 = row % n1;
                const int64_t i2 = (row / n1) % n2;
                const int64_t i3 = row / (n1 * n2);

                const float *__restrict x_row = x_base + i3 * xs3 + i2 * xs2 + i1 * xs1;
                const float *__restrict g_row = g_base + i3 * gs3 + i2 * gs2 + i1 * gs1;
                float *__restrict d_row       = dx_base + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                float sum_xx = 0.f, sum_xg = 0.f;
#ifndef GGML_SYCL_RMS_BACK_FAST
                float c_xx = 0.f, c_xg = 0.f;
#endif
                for (int64_t col = tid; col < D; col += WG) {
                    const float xv = x_row[col];
                    const float gv = g_row[col];
#ifdef GGML_SYCL_RMS_BACK_FAST
                    sum_xx += xv * xv;
                    sum_xg += xv * gv;
#else
                    float y1 = xv * xv - c_xx;
                    float t1 = sum_xx + y1;
                    c_xx = (t1 - sum_xx) - y1;
                    sum_xx = t1;

                    float y2 = xv * gv - c_xg;
                    float t2 = sum_xg + y2;
                    c_xg = (t2 - sum_xg) - y2;
                    sum_xg = t2;
#endif
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(sum_xx,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xx
#else
                    0.f
#endif
                );
                sycl::float2 xg = sycl::float2(sum_xg,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xg
#else
                    0.f
#endif
                );
                xx = warp_reduce_sum(xx, item_ct1);
                xg = warp_reduce_sum(xg, item_ct1);

                // cross-warp reduction using local memory (single barrier)
                const auto sub_group = item_ct1.get_sub_group();
                const auto sg_id     = sub_group.get_group_linear_id();
                const auto wi_in_sg  = sub_group.get_local_linear_id();
                const int nthreads   = item_ct1.get_local_range(2);
                const int nwarps     = nthreads / WARP_SIZE;

                sycl::float2 xx_total = xx;
                sycl::float2 xg_total = xg;
                if (nwarps > 1) {
                    if (wi_in_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);

                    if (sg_id == 0) {
                        const unsigned wi_u = wi_in_sg;
                        sycl::float2 xx_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_total = warp_reduce_sum(xx_first, item_ct1);
                        xg_total = warp_reduce_sum(xg_first, item_ct1);
                    } else {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_total = xx;
                        xg_total = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                float inv_r = 0.f;
                float coeff = 0.f;
                if (tid == 0) {
                    const float sum_xx_f  = xx_total.x() + xx_total.y();
                    const float sum_xdz_f = xg_total.x() + xg_total.y();
                    const float mean_eps  = sum_xx_f / (float) D + eps;
                    const float sum_eps   = sum_xx_f + eps * (float) D;
                    inv_r = sycl::rsqrt(mean_eps);
                    coeff = -sum_xdz_f / sum_eps;
                }
                inv_r = sycl::group_broadcast(item_ct1.get_group(), inv_r);
                coeff = sycl::group_broadcast(item_ct1.get_group(), coeff);

                for (int64_t col = tid; col < D; col += WG) {
                    d_row[col] = (g_row[col] + coeff * x_row[col]) * inv_r;
                }
            });
    });

    });
}

void ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    return norm_debug_call(__func__, ctx, { dst }, [&]() {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    const size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    const int64_t ss0 = nb00 / ts0;
    const int64_t ss1 = nb01 / ts0;
    const int64_t ss2 = nb02 / ts0;
    const int64_t ss3 = nb03 / ts0;
    const int64_t ds0 = nb0 / tdst;
    const int64_t ds1 = nb1 / tdst;
    const int64_t ds2 = nb2 / tdst;
    const int64_t ds3 = nb3 / tdst;

    /*support both WARP_SIZE or WARP_32_SIZE in code
      choose by hardware for better performance
    */
    l2_norm_f32_sycl<WARP_SIZE>(src0_d, dst_d, ne00, ne01, ne02, ne03,
            ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, stream, ctx.device);
    });
}

// nodes[0..count) are independent, same-shape, same-eps, same-nb L2_NORM ops validated
// by the caller; requires ncols < 1024 (the warp reduction path).
void ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, int count) {
    return norm_debug_call(__func__, ctx, {}, [&]() {
    const ggml_tensor * s0 = nodes[0]->src[0];
    const int ncols       = (int) s0->ne[0];
    const int nrows_total = (int) ggml_nrows(s0);
    float eps;
    memcpy(&eps, nodes[0]->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    for (int t = 0; t < count; ++t) {
        p.src[t] = (const float *) nodes[t]->src[0]->data;
        p.dst[t] = (float *) nodes[t]->data;
    }

    const ggml_tensor * d0 = nodes[0];
    const size_t        ts = ggml_type_size(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (int) s0->ne[1];
    st.ne2 = (int) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_SIZE>(p, st, count, ncols, nrows_total, eps, ctx.stream());
    }, nodes, count);
}
