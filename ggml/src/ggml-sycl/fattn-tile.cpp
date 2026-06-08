#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-tile.hpp"
#include <cmath>
#include <float.h>
namespace syclex = sycl::ext::oneapi::experimental;

void ggml_sycl_flash_attn_ext_tile_case(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, int DKQ, int DV) {
    const ggml_tensor * KQV = dst;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
    const bool use_logit_softcap = (logit_softcap != 0.0f);

    if (DKQ ==  40 && DV ==  40) {
        if (!use_logit_softcap) { launch_fattn_tile_switch_ncols2< 40,  40, false>(ctx, dst); return; }
        else                    { launch_fattn_tile_switch_ncols2< 40,  40, true >(ctx, dst); return; }
    }
    if (DKQ ==  64 && DV ==  64) {
        if (!use_logit_softcap) { launch_fattn_tile_switch_ncols2< 64,  64, false>(ctx, dst); return; }
        else                    { launch_fattn_tile_switch_ncols2< 64,  64, true >(ctx, dst); return; }
    }
    if (DKQ ==  72 && DV ==  72) {
        if (!use_logit_softcap) { launch_fattn_tile_switch_ncols2< 72,  72, false>(ctx, dst); return; }
        else                    { launch_fattn_tile_switch_ncols2< 72,  72, true >(ctx, dst); return; }
    }
    if (DKQ ==  80 && DV ==  80) {
        if (!use_logit_softcap) { launch_fattn_tile_switch_ncols2< 80,  80, false>(ctx, dst); return; }
        else                    { launch_fattn_tile_switch_ncols2< 80,  80, true >(ctx, dst); return; }
    }
    if (DKQ ==  96 && DV ==  96) {
        if (!use_logit_softcap) { launch_fattn_tile_switch_ncols2< 96,  96, false>(ctx, dst); return; }
        else                    { launch_fattn_tile_switch_ncols2< 96,  96, true >(ctx, dst); return; }
    }
    if (DKQ == 112 && DV == 112) {
        if (!use_logit_softcap) { launch_fattn_tile_switch_ncols2<112, 112, false>(ctx, dst); return; }
        else                    { launch_fattn_tile_switch_ncols2<112, 112, true >(ctx, dst); return; }
    }
    if (DKQ == 128 && DV == 128) {
        if (!use_logit_softcap) { launch_fattn_tile_switch_ncols2<128, 128, false>(ctx, dst); return; }
        else                    { launch_fattn_tile_switch_ncols2<128, 128, true >(ctx, dst); return; }
    }
    if (DKQ == 256 && DV == 256) {
        if (!use_logit_softcap) { launch_fattn_tile_switch_ncols2<256, 256, false>(ctx, dst); return; }
        else                    { launch_fattn_tile_switch_ncols2<256, 256, true >(ctx, dst); return; }
    }
    if (DKQ == 512 && DV == 512) {
        if (!use_logit_softcap) { launch_fattn_tile_switch_ncols2<512, 512, false>(ctx, dst); return; }
        else                    { launch_fattn_tile_switch_ncols2<512, 512, true >(ctx, dst); return; }
    }
    if (DKQ == 576 && DV == 512) {
        if (!use_logit_softcap) { launch_fattn_tile_switch_ncols2<576, 512, false>(ctx, dst); return; }
        else                    { launch_fattn_tile_switch_ncols2<576, 512, true >(ctx, dst); return; }
    }
    GGML_ABORT("Unsupported DKQ/DV combination");
}

void ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    switch (K->ne[0]) {
        case  40: { GGML_ASSERT(V->ne[0] == K->ne[0]); ggml_sycl_flash_attn_ext_tile_case(ctx, dst,  40,  40); } break;
        case  64: { GGML_ASSERT(V->ne[0] == K->ne[0]); ggml_sycl_flash_attn_ext_tile_case(ctx, dst,  64,  64); } break;
        case  72: { GGML_ASSERT(V->ne[0] == K->ne[0]); ggml_sycl_flash_attn_ext_tile_case(ctx, dst,  72,  72); } break;
        case  80: { GGML_ASSERT(V->ne[0] == K->ne[0]); ggml_sycl_flash_attn_ext_tile_case(ctx, dst,  80,  80); } break;
        case  96: { GGML_ASSERT(V->ne[0] == K->ne[0]); ggml_sycl_flash_attn_ext_tile_case(ctx, dst,  96,  96); } break;
        case 112: { GGML_ASSERT(V->ne[0] == K->ne[0]); ggml_sycl_flash_attn_ext_tile_case(ctx, dst, 112, 112); } break;
        case 128: { GGML_ASSERT(V->ne[0] == K->ne[0]); ggml_sycl_flash_attn_ext_tile_case(ctx, dst, 128, 128); } break;
        case 256: { GGML_ASSERT(V->ne[0] == K->ne[0]); ggml_sycl_flash_attn_ext_tile_case(ctx, dst, 256, 256); } break;
        case 512: { GGML_ASSERT(V->ne[0] == K->ne[0]); ggml_sycl_flash_attn_ext_tile_case(ctx, dst, 512, 512); } break;
        case 576: { GGML_ASSERT(V->ne[0] == 512);       ggml_sycl_flash_attn_ext_tile_case(ctx, dst, 576, 512); } break;
        default:  { GGML_ABORT("Unsupported head size"); } break;
    }
}
