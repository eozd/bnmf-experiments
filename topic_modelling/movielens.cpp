#include "bnmf_algs.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std::chrono;
using namespace bnmf_algs;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " filename n_components max_iter"
                  << std::endl;
        return -1;
    }
    const std::string filename(argv[1]);
    const size_t n_components = std::stoul(argv[2]);
    const size_t max_iter = std::stoul(argv[3]);

    std::ifstream datafile(filename);
    assert(datafile);
    constexpr size_t n_rows = 943, n_cols = 1682;

    matrix_t<double> X = matrix_t<double>::Constant(n_rows, n_cols, NAN);
    int value;
    size_t user_id, movie_id, timestamp;
    while (datafile >> user_id) {
        datafile >> movie_id >> value >> timestamp;
        X(user_id - 1, movie_id - 1) = value;
    }

    {
        std::ofstream x_orig_file("X_orig.txt", std::ios_base::trunc);
        x_orig_file << X;
    }

	double gamma = 1;
	std::vector<double> alpha(n_rows, 0.05);
	std::vector<double> beta(n_cols, 2);
	double a = 1000;
	double b = a / (5 * n_rows * n_cols);

/* 	std::vector<double> alpha(n_rows, 0.05); */
/* 	std::vector<double> beta(n_cols, 10); */
/* 	beta.back() = 60; */
/* 	double a = 100; */
/* 	double b = 1; */

	alloc_model::Params<double> params(a, b, alpha, beta);
	std::vector<alloc_model::Params<double>> param_vec(n_components, params);

    std::cout << "Computing the factorization" << std::endl;
    auto alg_begin_time = high_resolution_clock::now();

    // without psi approximate
    auto em_res = bld::online_EM(X, param_vec, max_iter, false);
	/* tensord<3> S = bld::bld_mult(X, n_components, params, max_iter, true); */

    auto alg_end_time = high_resolution_clock::now();
    std::cout
        << "Total time: "
        << duration_cast<milliseconds>(alg_end_time - alg_begin_time).count()
        << " milliseconds" << std::endl;

	matrixd logW = em_res.logW.array();
    matrixd logH = em_res.logH.array();
	matrixd S_ipk = em_res.S_ipk;
	matrixd S_pjk = em_res.S_pjk;
	/* tensord<2> S_ipk = S.sum(shape<1>({1})); */
	/* tensord<2> S_pjk = S.sum(shape<1>({0})); */
	/* Eigen::Map<matrixd> S_ipk_mat(S_ipk.data(), n_rows, n_components); */
	/* Eigen::Map<matrixd> S_pjk_mat(S_pjk.data(), n_cols, n_components); */


    std::ofstream x_file("X_full.txt", std::ios_base::trunc);
    std::ofstream h_file("logW.txt", std::ios_base::trunc);
    std::ofstream w_file("logH.txt", std::ios_base::trunc);
    std::ofstream s_ipk_file("S_ipk.txt", std::ios_base::trunc);
    std::ofstream s_pjk_file("S_pjk.txt", std::ios_base::trunc);

    x_file << em_res.X_full;
    w_file << logW;
    h_file << logH;
	s_ipk_file << S_ipk;
	s_pjk_file << S_pjk;
}
