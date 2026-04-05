#pragma once

#include "../cuda_compat.h"
#include "../cuda_utils.h" 
#include "../dispatch_utils.h"

#include "../ggml-common.h"
#include "../vecdotq.cuh"
#include "../mmq.cuh"

template <typename scalar_t, ggml_type type>
void mul_mat_q_case(const mmq_args<scalar_t> & args, cudaStream_t stream) {
    const cuda_device_info cuda_info = get_cuda_info();
    const int nsm   = cuda_info.nsm;
    const int cc    = cuda_info.cc;
    const size_t smpbo = cuda_info.smpbo;

    const int mmq_x_max = get_mmq_x_max_host(cc);
    const int mmq_y = get_mmq_y_host(cc);
    const int block_num_y = (args.ne01 + mmq_y - 1) / mmq_y;

    const bool use_stream_k = cc >= 700 && cc < 1000000;

    int mmq_x_best  = 0;
    int nparts_best = INT_MAX;

    for (int mmq_x = 8; mmq_x <= mmq_x_max && nparts_best > 1; mmq_x += 8) {
        const int granularity = mmq_get_granularity_host(mmq_x, cc);
        if (mmq_x % granularity != 0 || (size_t)mmq_get_shmem<type>(mmq_x, mmq_y, cc) > smpbo) {
            continue;
        }

        const int ntiles_x = (args.ne11 + mmq_x - 1) / mmq_x;
        const int nwaves_xy_tiling = ntiles_x * block_num_y;

        const int nparts = use_stream_k ? ntiles_x : nwaves_xy_tiling;

        if (nparts < nparts_best) {
            mmq_x_best  = mmq_x;
            nparts_best = nparts;
        }
    }

    switch (mmq_x_best) {
        case   8:
            launch_mul_mat_q<scalar_t, type,   8>(args, stream);
            break;
        case  16:
            launch_mul_mat_q<scalar_t, type,  16>(args, stream);
            break;
        case  24:
            launch_mul_mat_q<scalar_t, type,  24>(args, stream);
            break;
        case  32:
            launch_mul_mat_q<scalar_t, type,  32>(args, stream);
            break;
        case  40:
            launch_mul_mat_q<scalar_t, type,  40>(args, stream);
            break;
        case  48:
            launch_mul_mat_q<scalar_t, type,  48>(args, stream);
            break;
        case  56:
            launch_mul_mat_q<scalar_t, type,  56>(args, stream);
            break;
        case  64:
            launch_mul_mat_q<scalar_t, type,  64>(args, stream);
            break;
        case  72:
            launch_mul_mat_q<scalar_t, type,  72>(args, stream);
            break;
        case  80:
            launch_mul_mat_q<scalar_t, type,  80>(args, stream);
            break;
        case  88:
            launch_mul_mat_q<scalar_t, type,  88>(args, stream);
            break;
        case  96:
            launch_mul_mat_q<scalar_t, type,  96>(args, stream);
            break;
        case 104:
            launch_mul_mat_q<scalar_t, type, 104>(args, stream);
            break;
        case 112:
            launch_mul_mat_q<scalar_t, type, 112>(args, stream);
            break;
        case 120:
            launch_mul_mat_q<scalar_t, type, 120>(args, stream);
            break;
        case 128:
            launch_mul_mat_q<scalar_t, type, 128>(args, stream);
            break;
        default:
            assert(false);
            break;
    }
}
