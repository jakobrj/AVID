//
// Created by jakobrj on 3/8/21.
//

#ifndef PROCLUS_GPU_GPU_PROCLUS_SAVE_CUH
#define PROCLUS_GPU_GPU_PROCLUS_SAVE_CUH

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "curand_kernel.h"

class Result {
public:
	int n; int d;int k;
	int* d_M = nullptr;
	bool* d_D = nullptr;
	int* d_C = nullptr;
	int* h_M = nullptr;
	int* h_C = nullptr;
	bool* h_D = nullptr;
	Result() { n = 0;d = 0;k = 0; }
	Result(int n, int d, int k, int* d_M, bool* d_D, int* d_C) :n(n), d(d), k(k), d_M(d_M), d_D(d_D), d_C(d_C) {}

	int* get_h_M() {
		if (h_M == nullptr) {
			h_M = new int[k]();
			cudaMemcpy(h_M, d_M, k * sizeof(int), cudaMemcpyDeviceToHost);
		}
		return h_M;
	}
	int* get_h_C() {
		if (h_C == nullptr) {
			h_C = new int[n]();
			cudaMemcpy(h_C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);
		}
		return h_C;
	}
	bool* get_h_D() {
		if (h_D == nullptr) {
			h_D = new bool[k * d]();
			cudaMemcpy(h_D, d_D, k * d * sizeof(bool), cudaMemcpyDeviceToHost);
		}
		return h_D;
	}
};

Result
GPU_PROCLUS(float* d_data, int n, int d, int k, int l, float a, float b, float min_deviation, int termination_rounds, bool debug);

Result
GPU_PROCLUS_KEEP(float* d_data, int n, int d, int k, int l, float a, float b, float min_deviation, int termination_rounds, bool debug);

Result
GPU_PROCLUS_SAVE(float* d_data, int n, int d, int k, int l, float a, float b, float min_deviation, int termination_rounds, bool debug);

std::vector <Result>
GPU_PROCLUS_PARAM(float* d_data, int n, int d, std::vector<int> ks, std::vector<int> ls, float a, float b, float min_deviation,
	int termination_rounds);

std::vector <Result>
GPU_PROCLUS_PARAM_2(float* d_data, int n, int d, std::vector<int> ks, std::vector<int> ls, float a, float b, float min_deviation,
	int termination_rounds);

std::vector <Result>
GPU_PROCLUS_PARAM_3(float* d_data, int n, int d, std::vector<int> ks, std::vector<int> ls, float a, float b, float min_deviation,
	int termination_rounds);


class GPU_FAST_PROCLUS_C {

private:
	// arguments
	float* d_data;
	int n;
	int d;
	int k_max;
	float a;
	float b;
	float min_deviation;
	int termination_rounds;

	// values
	int Ak;
	int Bk;
	int previous_k = 0;

	// arrays
	curandState* d_state;
	bool* d_bad;
	int* d_C;
	int* d_C_sizes;
	int* d_C_best;
	int* d_C_sizes_best;
	int* d_C_result;
	float* d_cost;
	float* d_cost_best;
	bool* d_D;
	int* d_Ds;
	int* d_D_sizes;
	float* d_delta;
	float* d_delta_old;
	float* d_dist_n_Bk;
	bool* d_dist_n_Bk_set;
	float* d_H;
	int* d_L;
	int* d_L_sizes;
	int* d_L_sizes_change;
	int* d_lambda;
	int* d_lock;
	int* d_M;
	int* d_M_best;
	int* d_M_current;
	int* d_M_idx;
	int* d_M_idx_best;
	int* d_M_random;
	int* d_S;
	float* d_sigma;
	int* d_termination_criterion;
	float* d_X;
	float* d_Z;

	// results
	std::map<std::pair<int, int>, Result> results;

public:
	GPU_FAST_PROCLUS_C(float* d_data, int n, int d, int k_max, float a, float b, float min_deviation, int termination_rounds);

	Result get_result(int k, int l);

	Result compute(int k, int l);

	void clear_results();
};

#endif //PROCLUS_GPU_GPU_PROCLUS_SAVE_CUH
