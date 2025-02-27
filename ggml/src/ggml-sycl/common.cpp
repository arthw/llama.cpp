//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "common.hpp"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

int get_current_device_id() {
  return dpct::dev_mgr::instance().current_device_id();
}

void* ggml_sycl_host_malloc(size_t size) try {
  if (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    return nullptr;
  }

  void* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  auto q = dpct::get_in_order_queue();
//   sycl::queue q = *ggml_sycl_info().device_mgr->qptrs[0][0];

  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (void*)sycl::malloc_host(size, q));

  if (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    return nullptr;
  }

  return ptr;
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_sycl_host_free(void* ptr) try {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, dpct::get_in_order_queue())));
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static inline int get_sycl_env(const char *env_name, int default_val) {
    char *user_device_string = getenv(env_name);
    int user_number = default_val;

    unsigned n;
    if (user_device_string != NULL &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_number = (int)n;
    } else {
        user_number = default_val;
    }
    return user_number;
}

void print_device_detail_part1(int id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string version;
    version += std::to_string(prop.get_major_version());
    version += ".";
    version += std::to_string(prop.get_minor_version());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    auto global_mem_size = prop.get_global_mem_size()/1000000;

    fprintf(stderr, "|%2d|%19s|%5s|%39s|%14luM|\n", id, device_type.c_str(), version.c_str(),
        name.c_str(), global_mem_size);
}

void print_device_detail_part2(int id, sycl::device &device) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    fprintf(stderr, "|%2d|%17d|%14d|%12d|%34s|\n", id,
        prop.get_max_compute_units(),
        prop.get_max_work_group_size(), prop.get_max_sub_group_size(),
        device.get_info<sycl::info::device::driver_version>().c_str());
}

void print_device_opt_feature(ggml_sycl_device_info &info) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    int device_count = info.device_count;

    for (int id = 0; id < device_count; ++id) {
        sycl::device device = dpct::dev_mgr::instance().get_device(id);
        std::string backend_type = get_device_backend_and_type(device);
        int type_id = DeviceNums[backend_type]++;
        std::stringstream device_type;
        device_type << "[" << backend_type << ":" << std::to_string(type_id)
                    << "]";
        std::string device_type_s = device_type.str();
        device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
        GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
            info.infos[id].opt_feature.reorder ? "Y": "N");
    }
}

void ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    int device_count = dpct::dev_mgr::instance().device_count();
    std::map<std::string, size_t> DeviceNums;
    fprintf(stderr, "found %d SYCL devices:\n", device_count);
    fprintf(stderr, "Part1:\n");
    fprintf(stderr, "|ID|        Device Type|  Ver|                                   Name|Global mem size|\n");
    fprintf(stderr, "|--|-------------------|-----|---------------------------------------|---------------|\n");
    for (int id = 0; id < device_count; ++id) {
        sycl::device device = dpct::dev_mgr::instance().get_device(id);
        std::string backend_type = get_device_backend_and_type(device);
        int type_id=DeviceNums[backend_type]++;
        std::stringstream device_type;
        device_type << "[" <<  backend_type << ":" << std::to_string(type_id) << "]";
        print_device_detail_part1(id, device, device_type.str());
    }

    std::map<std::string, size_t> DeviceNums2;
    fprintf(stderr, "\nPart2:\n");
    fprintf(stderr, "|ID|Max compute units|Max work group|Max subgroup|                    Driver version|\n");
    fprintf(stderr, "|--|-----------------|--------------|------------|----------------------------------|\n");
    for (int id = 0; id < device_count; ++id) {
        sycl::device device = dpct::dev_mgr::instance().get_device(id);
        std::string backend_type = get_device_backend_and_type(device);
        int type_id=DeviceNums2[backend_type]++;
        std::stringstream device_type;
        device_type << "[" <<  backend_type << ":" << std::to_string(type_id) << "]";
        print_device_detail_part2(id, device);
    }
}

static ggml_sycl_device_info ggml_sycl_init(int main_gpu_id) try {
    static bool initialized = false;
    static ggml_sycl_device_info info(main_gpu_id);
    if (!initialized) {
        g_ggml_sycl_debug = get_sycl_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_disable_optimize= get_sycl_env("GGML_SYCL_DISABLE_OPT", 0);
        GGML_SYCL_DEBUG("[SYCL] call ggml_check_sycl\n");
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_OPT: %d\n", g_ggml_sycl_disable_optimize);
        GGML_LOG_INFO("Build with Macros:\n");
#if defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif
#if defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#endif

#if defined(SYCL_USE_XMX)
        GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
#else
        GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
#endif

        if (CHECK_TRY_ERROR(g_all_sycl_device_count =
                                dpct::dev_mgr::instance().device_count()) !=
            0) {
            initialized = true;
            GGML_LOG_INFO("  g_all_sycl_device_count is wrong:%d\n",
                g_all_sycl_device_count);
            return info;
        }
        GGML_ASSERT(g_all_sycl_device_count <= GGML_SYCL_MAX_DEVICES);

        if (info.device_count == 0) {
            GGML_LOG_INFO("%s: failed to initialize " GGML_SYCL_NAME ": no available device found\n",
                    __func__);
            return info;
        }
        GGML_ASSERT(info.device_count <= GGML_SYCL_MAX_DEVICES);

        ggml_backend_sycl_print_sycl_devices();
        print_device_opt_feature(info);
        initialized = true;
    }

    return info;

} catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

ggml_sycl_device_info &ggml_sycl_info(int main_gpu_id) {
    static ggml_sycl_device_info info = ggml_sycl_init(main_gpu_id);
    return info;
}

//--ggml_sycl_device_info--
bool gpu_has_xmx(sycl::device &dev) {
    return dev.has(sycl::aspect::ext_intel_matrix);
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_num, int64_t block_size) {
  const int64_t max_range = std::numeric_limits<int>::max();
  int64_t sycl_down_blk_size = block_size;
  int64_t global_range = accumulate_block_num * sycl_down_blk_size;
  while(global_range > max_range) {
      sycl_down_blk_size /= 2;
      global_range = accumulate_block_num * sycl_down_blk_size;
  }
  return sycl_down_blk_size;
}

void ggml_sycl_op_flatten(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 const ggml_sycl_op_flatten_t op) try {

    const bool use_src1 = src1 != nullptr;
    if(use_src1)
      GGML_ASSERT(strcmp(src1->buffer->buft->iface.get_name(src1->buffer->buft), GGML_SYCL_NAME "_Split") != 0);
    GGML_ASSERT(strcmp(dst->buffer->buft->iface.get_name(dst->buffer->buft), GGML_SYCL_NAME "_Split") != 0);

    // dd = data device
    float * src0_ddf = (float *) src0->data;
    float * src1_ddf = use_src1 ? (float *) src1->data : nullptr;
    float *  dst_ddf = (float *) dst->data;

    ggml_sycl_pool_alloc<float> src0_f(ctx.pool());
    ggml_sycl_pool_alloc<float> src1_f(ctx.pool());
    ggml_sycl_pool_alloc<float>  dst_f(ctx.pool());

    ggml_sycl_set_device(ctx.device);
    queue_ptr main_stream = ctx.stream();
    // GGML_SYCL_DEBUG("ctx.device=%d, main_stream=%p src0_on_device=%d, src1_on_device=%d, dst_on_device=%d\n",
        // ctx.device, main_stream, src0_on_device, src1_on_device, dst_on_device);

    // do the computation
    op(ctx, src0, src1, dst, src0_ddf, src1_ddf, dst_ddf, main_stream);
    // print_ggml_tensor("tensor", dst);
}
catch (sycl::exception const &exc) {

  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


void release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        for (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            if (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        if (extra->data_device[i] != nullptr && streams.size()>0) {
            ggml_sycl_set_device(i);
            SYCL_CHECK(
                CHECK_TRY_ERROR(sycl::free(extra->data_device[i], *(streams[i]))));
        }
    }
    delete extra;
}
