//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//


#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-tile.hpp"
#include "fattn-vec.hpp"
#include "fattn.hpp"


static void ggml_sycl_flash_attn_ext_vec_case_impl(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst,
    int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap) {

    constexpr int    warp_size     = WARP_16_SIZE;
    const int        cc            = ggml_sycl_info().devices[ggml_sycl_get_device()].cc;
    const int        nthreads      = ggml_sycl_fattn_vec_get_nthreads_host(cc);
    const int        nwarps        = nthreads / warp_size;
    const bool       need_f16_K    = type_K == GGML_TYPE_F16;
    const bool       need_f16_V    = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;

    switch (D) {
        case 64:
            switch (cols_per_block) {
                case 1: {
#ifdef GGML_SYCL_FA_ALL_QUANTS
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q8_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q8_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
#else
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 1, 1, flash_attn_ext_vec<64, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
#endif
                } break;
                case 2: {
#ifdef GGML_SYCL_FA_ALL_QUANTS
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_F16, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_F16, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_F16, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_F16, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_F16, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_F16, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_F16, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_F16, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_F16, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_F16, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q8_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q8_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
#else
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<64, 2, 1, flash_attn_ext_vec<64, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 64, need_f16_K, need_f16_V, false); }
                        return;
                    }
#endif
                } break;
                default: break;
            }
            break;
        case 128:
            switch (cols_per_block) {
                case 1: {
#ifdef GGML_SYCL_FA_ALL_QUANTS
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
#else
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 1, 1, flash_attn_ext_vec<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
#endif
                } break;
                case 2: {
#ifdef GGML_SYCL_FA_ALL_QUANTS
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_F16, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_F16, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_F16, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_F16, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_F16, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_F16, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_F16, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_F16, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_F16, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_F16, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
#else
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<128, 2, 1, flash_attn_ext_vec<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 128, need_f16_K, need_f16_V, false); }
                        return;
                    }
#endif
                } break;
                default: break;
            }
            break;
        case 256:
            switch (cols_per_block) {
                case 1: {
#ifdef GGML_SYCL_FA_ALL_QUANTS
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
#else
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 1, 1, flash_attn_ext_vec<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
#endif
                } break;
                case 2: {
#ifdef GGML_SYCL_FA_ALL_QUANTS
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_F16, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_F16, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_F16, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_F16, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_F16, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_F16, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_F16, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_F16, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_F16, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_F16, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
#else
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<256, 2, 1, flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 256, need_f16_K, need_f16_V, false); }
                        return;
                    }
#endif
                } break;
                default: break;
            }
            break;
        case 512:
            switch (cols_per_block) {
                case 1: {
#ifdef GGML_SYCL_FA_ALL_QUANTS
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_F16, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_F16, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_F16, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_F16, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_F16, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_F16, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_F16, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_F16, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_F16, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_F16, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q8_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q8_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
#else
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 1, 1, flash_attn_ext_vec<512, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
#endif
                } break;
                case 2: {
#ifdef GGML_SYCL_FA_ALL_QUANTS
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_F16, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_F16, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_F16, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_F16, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_F16, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_F16, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_F16, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_F16, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_F16, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_F16, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_1, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_1, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q8_0, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q8_0, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_1) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
#else
                    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_F16, GGML_TYPE_F16, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_F16, GGML_TYPE_F16, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
                    if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
                        if (use_logit_softcap) { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false);  }
                        else                  { launch_fattn<512, 2, 1, flash_attn_ext_vec<512, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, warp_size>, warp_size>(ctx, dst, nwarps, nbytes_shared, 512, need_f16_K, need_f16_V, false); }
                        return;
                    }
#endif
                } break;
                default: break;
            }
            break;
        default:
            break;
    }
    GGML_ABORT("Unsupported combination in ggml_sycl_flash_attn_ext_vec_case_impl");
}

void ggml_sycl_flash_attn_ext_vec_case(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst,
    int D, ggml_type type_K, ggml_type type_V) {

    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    const int  cols_per_block    = (Q->ne[1] == 1) ? 1 : 2;
    const bool use_logit_softcap = (logit_softcap != 0.0f);

    ggml_sycl_flash_attn_ext_vec_case_impl(ctx, dst, D, cols_per_block, type_K, type_V, use_logit_softcap);
}

static void ggml_sycl_flash_attn_ext_vec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

    // F32 tensors are handled as F16 for flash attention
    const ggml_type type_K = (K->type == GGML_TYPE_F32) ? GGML_TYPE_F16 : K->type;
    const ggml_type type_V = (V->type == GGML_TYPE_F32) ? GGML_TYPE_F16 : V->type;

    switch (Q->ne[0]) {
        case 64: {
#ifdef GGML_SYCL_FA_ALL_QUANTS
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_F16, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q4_0, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q4_1, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q5_0, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q5_1, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q8_0, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_F16, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_F16, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_F16, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_F16, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_F16, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0); return; }
#else
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_F16, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 64, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0); return; }
#endif
        } break;
        case 128: {
#ifdef GGML_SYCL_FA_ALL_QUANTS
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_F16, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q4_0, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q4_1, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q5_0, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q5_1, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q8_0, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_F16, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_F16, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_F16, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_F16, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_F16, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0); return; }
#else
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_F16, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0); return; }
#endif
        } break;
        case 256: {
#ifdef GGML_SYCL_FA_ALL_QUANTS
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_F16, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q4_0, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q4_1, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q5_0, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q5_1, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q8_0, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_F16, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_F16, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_F16, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_F16, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_F16, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0); return; }
#else
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_F16, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0); return; }
#endif
        } break;
        case 512: {
#ifdef GGML_SYCL_FA_ALL_QUANTS
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_F16, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q4_0, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q4_1, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q5_0, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q5_1, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q8_0, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_F16, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_F16, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_F16, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_F16, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q5_1) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1); return; }
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_F16, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q4_1 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q5_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q5_1 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0); return; }
#else
            if (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_F16)  { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_F16, GGML_TYPE_F16); return; }
            if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0); return; }
            if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) { ggml_sycl_flash_attn_ext_vec_case(ctx, dst, 512, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0); return; }
#endif
        } break;
        default:
            break;
    }

    GGML_ABORT("Not match KV type in vec");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_TILE     = 200,
};

static best_fattn_kernel ggml_sycl_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
#ifndef SYCL_FLASH_ATTN
    GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// SYCL_FLASH_ATTN

    if(!g_ggml_sycl_enable_flash_attention) return BEST_FATTN_KERNEL_NONE;

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_SYCL_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_SYCL_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_SYCL_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_SYCL_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    const bool can_use_vector_kernel = Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // Todo: Use the XMX kernel if possible:

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_set_device(ctx.device);
    switch (ggml_sycl_get_best_fattn_kernel(ggml_sycl_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("Not support Flash-Attention");
        case BEST_FATTN_KERNEL_TILE:
            ggml_sycl_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_sycl_flash_attn_ext_vec(ctx, dst);
            break;
    }
}

bool ggml_sycl_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_sycl_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
