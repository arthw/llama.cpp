#include "sycl_hw.hpp"


sycl_hw_info get_device_hw_info(sycl::device *device_ptr) {
    sycl_hw_info res;
    int32_t id = device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
    res.device_id = id;

    syclex::architecture arch = device_ptr->get_info<syclex::info::device::architecture>();
    res.arch = arch;

    res.family = get_device_family(arch);

    return res;
}


SYCL_HW_FAMILY get_device_family(syclex::architecture arch) {
    if (arch == syclex::architecture::intel_gpu_dg1 ||
        arch == syclex::architecture::intel_gpu_acm_g10 ||
        arch == syclex::architecture::intel_gpu_acm_g11 ||
        arch == syclex::architecture::intel_gpu_acm_g12 ||
        arch == syclex::architecture::intel_gpu_pvc ||
        arch == syclex::architecture::intel_gpu_pvc_vg ||
        arch == syclex::architecture::intel_gpu_bmg_g21
        ) {
        return SYCL_HW_FAMILY_INTEL_dGPU;
    } else if (
        arch == syclex::architecture::intel_gpu_mtl_u ||
        arch == syclex::architecture::intel_gpu_mtl_s ||
        arch == syclex::architecture::intel_gpu_mtl_h ||
        arch == syclex::architecture::intel_gpu_arl_u ||
        arch == syclex::architecture::intel_gpu_arl_s ||
        arch == syclex::architecture::intel_gpu_arl_h ||
        arch == syclex::architecture::intel_gpu_lnl_m
       ) {
        return SYCL_HW_FAMILY_INTEL_BUILT_IN_GPU;
    } else {
        return SYCL_HW_FAMILY_INTEL_IGPU;
    }
}