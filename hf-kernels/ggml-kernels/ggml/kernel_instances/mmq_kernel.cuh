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
    const int nsm   = ggml_cuda_info().devices[id].nsm;
    const int cc    = ggml_cuda_info().devices[id].cc;
    const int smpbo = ggml_cuda_info().devices[id].smpbo;

    const int mmq_x_max = get_mmq_x_max_host(cc);
    const int mmq_y = get_mmq_y_host(cc, mmq_x_max);
    const int block_num_y = (args.ne01 + mmq_y - 1) / mmq_y;

    int mmq_x_best  = 0;
    int nwaves_best = INT_MAX;

    for (int mmq_x = 8; mmq_x <= mmq_x_max && nwaves_best > 1; mmq_x += 8) {
        const int block_num_x = (args.ne11 + mmq_x - 1) / mmq_x;
        const int nwaves = (block_num_x*block_num_y + nsm - 1) / nsm;

        if (nwaves < nwaves_best && mmq_get_shmem(type, mmq_x, mmq_y) <= smpbo) {
            mmq_x_best  = mmq_x;
            nwaves_best = nwaves;
        }
    }

    switch (mmq_x_best) {
        case   8:
            launch_mul_mat_q<type,   8, mmq_get_nwarps(  8)>(args, stream);
            break;
        case  16:
            launch_mul_mat_q<type,  16, mmq_get_nwarps( 16)>(args, stream);
            break;
        case  24:
            launch_mul_mat_q<type,  24, mmq_get_nwarps( 24)>(args, stream);
            break;
        case  32:
            launch_mul_mat_q<type,  32, mmq_get_nwarps( 32)>(args, stream);
            break;
        case  40:
            launch_mul_mat_q<type,  40, mmq_get_nwarps( 40)>(args, stream);
            break;
        case  48:
            launch_mul_mat_q<type,  48, mmq_get_nwarps( 48)>(args, stream);
            break;
        case  56:
            launch_mul_mat_q<type,  56, mmq_get_nwarps( 56)>(args, stream);
            break;
        case  64:
            launch_mul_mat_q<type,  64, mmq_get_nwarps( 64)>(args, stream);
            break;
        case  72:
            launch_mul_mat_q<type,  72, mmq_get_nwarps( 72)>(args, stream);
            break;
        case  80:
            launch_mul_mat_q<type,  80, mmq_get_nwarps( 80)>(args, stream);
            break;
        case  88:
            launch_mul_mat_q<type,  88, mmq_get_nwarps( 88)>(args, stream);
            break;
        case  96:
            launch_mul_mat_q<type,  96, mmq_get_nwarps( 96)>(args, stream);
            break;
        case 104:
            launch_mul_mat_q<type, 104, mmq_get_nwarps(104)>(args, stream);
            break;
        case 112:
            launch_mul_mat_q<type, 112, mmq_get_nwarps(112)>(args, stream);
            break;
        case 120:
            launch_mul_mat_q<type, 120, mmq_get_nwarps(120)>(args, stream);
            break;
        case 128:
            launch_mul_mat_q<type, 128, mmq_get_nwarps(128)>(args, stream);
            break;
        default:
            assert(false);
            break;
    }
}