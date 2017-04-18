#include "partB.hpp"
#include <boost/filesystem.hpp>
int main() {

	// Maximum order of mixands to consider
	unsigned int M_min = 2;
	unsigned int M_max = 20;

	// Maximum order of mixands to consider
	unsigned int N_iter_max = 300;

	// Tolerance on icll
	double tol = 1e-3;

	// The training data is loaded
	std::cout << "Loading training data" << std::endl;
	arma::mat Xbar = arma::mat();
	arma::mat Ybar = arma::mat();

	Xbar. load("../../../data/training/Xbar_R.txt");
	Ybar. load("../../../data/training/Ybar_R.txt");

	std::cout << Xbar.n_rows << std::endl;

	unsigned int N_R = Ybar.n_cols;

	arma::mat bic = arma::zeros<arma::mat>(2, M_max - M_min + 1);


	std::cout << "Training models" << std::endl;

	for (unsigned int M = M_min ; M < M_max + 1 ; ++ M) {


		arma::vec omega = arma::vec(M);

		arma::mat Nu = arma::mat(8, M);
		arma::mat Mu = arma::ones<arma::mat>(175, M);
		arma::mat Gamma = arma::mat(M, N_R);

		std::vector<arma::mat> Sigma;
		std::vector<arma::mat> Psi;
		std::vector<arma::mat> Lambda;

		for (unsigned int m = 0; m < M; ++m) {
			Sigma.push_back(arma::eye<arma::mat>(8, 8));
			Psi.push_back(arma::eye<arma::mat>(175, 175));
			Lambda.push_back(arma::zeros<arma::mat>(175, 8));
		}

		std::cout << "Model comprised of : " << M << " mixands" << std::endl;

		init(M, Xbar, Ybar, omega, Nu, Sigma, Lambda, Mu, Psi, Gamma);

		double icll_old = ICLL(Xbar, Ybar, omega, Nu, Sigma, Lambda, Mu, Psi);
		double icll = 0;

		for (unsigned int n = 0; n < N_iter_max; ++n) {

			std::cout << "########## Iteration " << std::to_string(n + 1) << " ############" << std::endl;

			// E-step
			if (n > 0) {
				std::cout << "\tE-step" << std::endl;
				E_step(Xbar, Ybar, omega, Nu, Sigma, Lambda, Mu, Psi, Gamma);
			}


			// M-step
			std::cout << "\tM-step" << std::endl;
			M_step(Xbar, Ybar, omega, Nu, Sigma, Lambda, Mu, Psi, Gamma);

			// The icll is recomputed
			icll = ICLL(Xbar, Ybar, omega, Nu, Sigma, Lambda, Mu, Psi);

			std::cout << "ICLL: " << icll << std::endl;

			double change = (icll - icll_old) / std::abs(icll_old) * 100;
			std::cout << "Relative change in ICLL (%) " << change << " %" << std::endl;

			if (change < tol) {
				break;
			}

			icll_old = icll;

		}

		// The BIC scored is computed and stored
		double bic_s = bic_score(N_R, icll, M);
		arma::vec result = {M, bic_s};
		bic.col(M - M_min) = result;

		// I should save the model somewhere here

		std::string dir("../models/M_" + std::to_string(M) + "/" );

		omega.save(dir + "omega.mat", arma::raw_ascii);
		Nu.save(dir + "Nu.mat", arma::raw_ascii);
		Mu.save(dir + "Mu.mat", arma::raw_ascii);

		for (unsigned int m = 0; m < M; ++m) {
			Sigma[m].save(dir + "Sigma_" + std::to_string(m) + ".mat", arma::raw_ascii);
			Lambda[m].save(dir + "Lambda_" + std::to_string(m) + ".mat", arma::raw_ascii);
			Psi[m].save(dir + "Psi_" + std::to_string(m) + ".mat", arma::raw_ascii);
		}

	}

	// The bic score is saved
	bic.save("../models/bic.mat", arma::raw_ascii);

	return 0;
}