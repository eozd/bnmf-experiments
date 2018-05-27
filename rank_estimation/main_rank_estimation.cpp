#include "bnmf_algs.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>

using namespace std::chrono;
using namespace bnmf_algs;

/**
 * @brief Read a matrix X from a given file, run a factorization algorithm on it,
 * and write the resulting two matrices to W.txt and H.txt, and the resulting
 * tensor to S.txt files. If NMF is run, only W.txt and H.txt contain meaningful
 * results.
 *
 * Given input file must contain each row of the matrix on a separated line.
 * Each entry in a row must be separated using a single space character.
 */
int main(int argc, char** argv) {
    // TODO: Improve file parsing code to allow scientific notation and multiple
    if (argc != 4) {
        std::cout << "usage: " << argv[0]
                  << " filename n_components gamma"
                  << std::endl;
        return -1;
    }
    std::string filename(argv[1]);
    size_t n_components = std::stoul(argv[2]);
    double gamma = std::stod(argv[3]);

    std::ifstream datafile(filename);
    std::vector<double> data;
    int n_rows = 0;
    std::string line;
    double value;
    while (std::getline(datafile, line)) {
        size_t index = 0, prev_index = 0;
        index = line.find(' ');
        while (prev_index != std::string::npos) {
            value =
                std::atof(line.substr(prev_index, index - prev_index).data());
            data.push_back(value);

            prev_index = index;
            index = line.find(' ', index + 1);
        }
        ++n_rows;
    }
    auto n_cols = data.size() / n_rows;

    matrixd X = Eigen::Map<matrixd>(data.data(), n_rows, n_cols);
	double sum_x = X.sum();
	
    std::vector<double> alpha_dirichlet(X.rows(), (gamma * n_cols) / n_components);
	double a = n_rows * n_cols * gamma;
	double b = a / (X.mean() * n_rows * n_cols);

    std::vector<double> beta_dirichlet(n_components, (gamma * n_rows) / n_components);
    alloc_model::Params<double> params(a, b, alpha_dirichlet, beta_dirichlet);

	std::cout << alloc_model::total_log_marginal(X, params) << std::endl;
}
