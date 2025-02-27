#ifndef SYCL_HW_HPP
#define SYCL_HW_HPP

#include <algorithm>
#include <stdio.h>
#include <vector>
#include <map>

#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

enum SYCL_HW_FAMILY {
  SYCL_HW_FAMILY_UNKNOWN = -1,
  SYCL_HW_FAMILY_INTEL_IGPU = 0,
  SYCL_HW_FAMILY_INTEL_BUILT_IN_GPU = 1, //since MTL
  SYCL_HW_FAMILY_INTEL_dGPU = 2,
};

struct sycl_hw_info {
  syclex::architecture arch;
  int32_t device_id;
  SYCL_HW_FAMILY family;
};

SYCL_HW_FAMILY get_device_family(syclex::architecture arch);

sycl_hw_info get_device_hw_info(sycl::device *device_ptr);


#endif // SYCL_HW_HPP
