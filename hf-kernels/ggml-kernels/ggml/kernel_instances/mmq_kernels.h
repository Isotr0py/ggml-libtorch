#pragma once

#include "../mmq.cuh"

// Forward declarations for mul_mat_q_case functions
template <typename scalar_t>
void mul_mat_q_case_Q4_0(const mmq_args<scalar_t> & args, cudaStream_t stream);

template <typename scalar_t>
void mul_mat_q_case_Q4_1(const mmq_args<scalar_t> & args, cudaStream_t stream);

template <typename scalar_t>
void mul_mat_q_case_Q5_0(const mmq_args<scalar_t> & args, cudaStream_t stream);

template <typename scalar_t>
void mul_mat_q_case_Q5_1(const mmq_args<scalar_t> & args, cudaStream_t stream);

template <typename scalar_t>
void mul_mat_q_case_Q8_0(const mmq_args<scalar_t> & args, cudaStream_t stream);

template <typename scalar_t>
void mul_mat_q_case_Q2_K(const mmq_args<scalar_t> & args, cudaStream_t stream);

template <typename scalar_t>
void mul_mat_q_case_Q3_K(const mmq_args<scalar_t> & args, cudaStream_t stream);

template <typename scalar_t>
void mul_mat_q_case_Q4_K(const mmq_args<scalar_t> & args, cudaStream_t stream);

template <typename scalar_t>
void mul_mat_q_case_Q5_K(const mmq_args<scalar_t> & args, cudaStream_t stream);

template <typename scalar_t>
void mul_mat_q_case_Q6_K(const mmq_args<scalar_t> & args, cudaStream_t stream);