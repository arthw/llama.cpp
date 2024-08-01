#ifndef SYCL_DEVICE_HPP
#define SYCL_DEVICE_HPP

#include <algorithm>
#include <stdio.h>
#include <vector>


#include <sycl/sycl.hpp>
#include "dpct.hpp"

#include "ggml-sycl.h"
#include "presets.hpp"
// #include "common.hpp"

enum ggml_sycl_backend_device_filter {
  SYCL_ALL_DEVICES = 0,
  SYCL_DEVICES_TOP_LEVEL_ZERO,
  SYCL_VISIBLE_DEVICES
};

enum ggml_sycl_backend_gpu_mode {
    SYCL_UNSET_GPU_MODE = -1,
    SYCL_SINGLE_GPU_MODE = 0,
    SYCL_MUL_GPU_MODE
};


struct sycl_device_info {
    int     cc;                 // compute capability
    // int     nsm;                // number of streaming multiprocessors
    // size_t  smpb;               // max. shared memory per block
    bool    vmm;                // virtual memory support
    size_t  total_vram;

    int id;
    sycl::device device;
    int max_compute_units;
    int max_work_group_sizes;
    int hw_family;
    sycl::context ctx;
    sycl::queue * qptrs[GGML_SYCL_MAX_STREAMS] = { nullptr };
};

struct ggml_sycl_device_info {
    int device_mode = SYCL_MUL_GPU_MODE;
    int device_count;
    bool oneapi_device_selector_existed = false;
    bool sycl_visible_devices_existed = false;
    std::vector<int> ids;
    std::vector<sycl::device> devices;
    sycl::queue *first_queue;
    std::string device_list;
    sycl::context co_ctx;
    int m_device_filter;

    sycl_device_info infos[GGML_SYCL_MAX_DEVICES];
    std::array<float, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    ggml_sycl_device_info(int main_gpu_id);//single device mode

    void init(ggml_sycl_backend_device_filter device_filter);
    void init_single_mode(int main_gpu_id);

    void clear_infos();
    void print_gpu_device_list();
    int work_group_size(int device_id);
    bool is_allowed_device(int device_id);
    const char* devices_list();
    int get_device_id(int device_index);
    int hw_family(int device_id);

    sycl::queue *_create_queue_ptr(sycl::device device); //internal API to hide dpct API.
    void create_context_for_group_gpus();
    sycl::queue *create_queue_for_device(sycl::device &device);
    sycl::queue *create_queue_for_device_id(int device_id);
    int get_device_index(int device_id);
    void create_context_for_devices();
    void set_allow_devices();
    void detect_all_sycl_device_list();
    void detect_sycl_visible_device_list();
    void detect_sycl_gpu_list_with_max_cu();
    int get_device_count();
    bool is_ext_oneapi_device(const sycl::device &dev);
    void add_device_info(int id);
    void create_queues(int id);
    void create_queues_for_devices();
    std::vector<sycl::device> get_devices();
    std::vector<int> get_sycl_visible_devices();
    void update_mem();
    void init_devices_dynamic_info();

    sycl::context &get_co_ctx() { return co_ctx; }

};

static inline bool env_existed(const char *env_name);

#endif // SYCL_DEVICE_HPP
