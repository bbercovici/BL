#include "partB.hpp"
#include <boost/filesystem.hpp>
#include <string>

int main() {

	// Maximum order of mixands to consider
	unsigned int M_min = 2;
	unsigned int M_max = 20;

	// Maximum order of mixands to consider
	unsigned int N_iter_max = 300;

	// Tolerance on icll
	double tol = 1e-3;

	// // The training data is loaded
	std::cout << "Loading training data" << std::endl;
	arma::mat X_bar_R = arma::mat();
	arma::mat X_bar_Q = arma::mat();
	arma::mat Y_bar_R = arma::mat();


	X_bar_R. load("../../../data/training/Xbar_R.txt");
	X_bar_Q. load("../../../data/training/Xbar_Q.txt");

	Y_bar_R. load("../../../data/training/Ybar_R.txt");


	unsigned int N_R = Y_bar_R.n_cols;

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

		init(M, X_bar_R, Y_bar_R, omega, Nu, Sigma, Lambda, Mu, Psi, Gamma);

		double icll_old = ICLL(X_bar_R, Y_bar_R, omega, Nu, Sigma, Lambda, Mu, Psi);
		double icll = 0;

		for (unsigned int n = 0; n < N_iter_max; ++n) {

			std::cout << "########## Iteration " << std::to_string(n + 1) << " ############" << std::endl;

			// E-step
			if (n > 0) {
				std::cout << "\tE-step" << std::endl;
				E_step(X_bar_R, Y_bar_R, omega, Nu, Sigma, Lambda, Mu, Psi, Gamma);

			}


			// M-step
			std::cout << "\tM-step" << std::endl;
			M_step(X_bar_R, Y_bar_R, omega, Nu, Sigma, Lambda, Mu, Psi, Gamma);
				std::cout << omega << std::endl;

				std::cout << Nu.col(0) << std::endl;
				throw (std::runtime_error(""));

			// The icll is recomputed
			icll = ICLL(X_bar_R, Y_bar_R, omega, Nu, Sigma, Lambda, Mu, Psi);

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
		arma::vec result = {(double)(M), bic_s};
		bic.col(M - M_min) = result;

		// I should save the model somewhere here

		std::string dir("../models/M_" + std::to_string(M) + "/" );


		save_model(dir,
		           omega,
		           Nu,
		           Sigma ,
		           Lambda,
		           Mu,
		           Psi);






	}

	// The bic score is saved
	bic.save("../models/bic.mat", arma::raw_ascii);

	arma::vec omega;
	arma::mat Nu;
	arma::mat Mu;

	std::vector<arma::mat> Sigma;
	std::vector<arma::mat> Psi;
	std::vector<arma::mat> Lambda;

	std::string dir("../models/M_10/" );


	load_model(dir,
	           omega,
	           Nu,
	           Sigma ,
	           Lambda,
	           Mu,
	           Psi);



	std::cout << "Generating missing observations" << std::endl;
	arma::mat Y_bar_Q = XQ_to_YQ_GM( X_bar_Q,
	                                 omega,
	                                 Nu,
	                                 Sigma ,
	                                 Lambda,
	                                 Mu,
	                                 Psi);


	Y_bar_Q.save("../../../data/training/Ybar_Q.txt", arma::raw_ascii);















	return 0;
}