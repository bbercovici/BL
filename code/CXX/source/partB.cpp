#include "partB.hpp"



void init(unsigned int M,
          arma::mat & Xbar,
          arma::mat & Ybar,
          arma::vec & omega,
          arma::mat & Nu,
          std::vector<arma::mat> & Sigma ,
          std::vector<arma::mat> & Lambda,
          arma::mat & Mu,
          std::vector<arma::mat> & Psi,
          arma::mat & Gamma) {

	double N_R = Ybar.n_cols;

	arma::kmeans( Nu, Ybar, M, arma::random_spread, 500, true);

	// Arbitrary Gamma
	Gamma.fill(1. / M);

	// omega
	omega = arma::sum(Gamma, 1) / N_R;



}

double ICLL(arma::mat & Xbar,
            arma::mat & Ybar,
            arma::vec & omega,
            arma::mat & Nu,
            std::vector<arma::mat> & Sigma ,
            std::vector<arma::mat> & Lambda,
            arma::mat & Mu,
            std::vector<arma::mat> & Psi) {


	unsigned int N_R = Ybar.n_cols;
	unsigned int M = omega.n_rows;

	arma::mat log_Nx = arma::zeros<arma::mat>(N_R, M);
	arma::mat log_Ny = arma::zeros<arma::mat>(N_R, M);
	arma::mat mean_X;


	for (unsigned int m = 0 ; m < M; ++m ) {

		mean_X = Lambda[m] * Ybar;
		mean_X.each_col() += Mu.col(m);


		log_Nx.col(m) = log_gaussian_pdf_vec_X(Xbar, mean_X, Psi[m]);
		log_Ny.col(m) = log_gaussian_pdf_vec_general(Ybar, Nu.col(m), Sigma[m]);

	}

	arma::mat normalizers = arma::mat(2, N_R);

	normalizers.row(0) = arma::max(log_Nx, 1).t();
	normalizers.row(1) = arma::max(log_Ny, 1).t();

	arma::vec normalizer = arma::min(normalizers, 0).t();

	arma::vec partial_sum = arma::zeros<arma::vec>(N_R);

	for (unsigned int m = 0 ; m < M; ++m ) {
		partial_sum += arma::exp(std::log(omega(m)) + log_Nx.col(m) + log_Ny.col(m) - normalizer);
	}

	double icll = arma::sum(arma::log(partial_sum) + normalizer);

	return icll;
}


arma::vec log_gaussian_pdf_vec_general(arma::mat & Ybar,
                                       arma::vec mean,
                                       arma::mat & cov) {


	arma::vec log_gaussian = arma::vec(Ybar.n_cols);

	double log_det = std::log(arma::det(2 * arma::datum::pi * cov));

	#pragma omp parallel for
	for (unsigned int i = 0; i < Ybar.n_cols; ++ i) {

		log_gaussian(i) = - 0.5 * arma::dot(Ybar.col(i) - mean, arma::inv(cov) * (Ybar.col(i) - mean)) - 0.5 * log_det;
	}


	return log_gaussian;

}



arma::vec log_gaussian_pdf_vec_X(arma::mat & Xbar,
                                 arma::mat & mean,
                                 arma::mat & cov) {


	arma::vec log_gaussian = arma::vec(Xbar.n_cols);

	double log_det = arma::sum(arma::log(2 * arma::datum::pi * cov.diag()));

	#pragma omp parallel for
	for (unsigned int i = 0; i < Xbar.n_cols; ++ i) {

		log_gaussian(i) = - 0.5 * arma::dot(Xbar.col(i) - mean.col(i), arma::diagmat(1. / arma::diagmat(cov)) * (Xbar.col(i) - mean.col(i))) - 0.5 * log_det;
	}


	return log_gaussian;

}





void M_step(arma::mat & Xbar,
            arma::mat & Ybar,
            arma::vec & omega,
            arma::mat & Nu,
            std::vector<arma::mat> & Sigma ,
            std::vector<arma::mat> & Lambda,
            arma::mat & Mu,
            std::vector<arma::mat> & Psi,
            arma::mat & Gamma) {


	unsigned int N_R = Ybar.n_cols;
	unsigned int M = omega.n_rows;



	boost::progress_display show_progress(M);

	for (unsigned int m = 0; m < M; ++m) {
		Psi[m] = Psi_update(m, Xbar, Ybar, Lambda, Mu, Gamma);
		Sigma[m] = Sigma_update(m, Ybar, Nu, Gamma);
		Lambda[m] = Lambda_update(m, Xbar, Ybar, Mu, Gamma);

		Nu.col(m) = Nu_update(m, Ybar, Gamma);
		throw (std::runtime_error(""));

		Mu.col(m) = Mu_update(m, Xbar, Ybar, Lambda, Gamma);

		++show_progress;
	}
	omega = arma::sum(Gamma, 1) / N_R;




}

void E_step(arma::mat & Xbar,
            arma::mat & Ybar,
            arma::vec & omega,
            arma::mat & Nu,
            std::vector<arma::mat> & Sigma ,
            std::vector<arma::mat> & Lambda,
            arma::mat & Mu,
            std::vector<arma::mat> & Psi,
            arma::mat & Gamma) {


	unsigned int N_R = Ybar.n_cols;
	unsigned int M = omega.n_rows;

	arma::mat log_Nx = arma::zeros<arma::mat>(N_R, M);
	arma::mat log_Ny = arma::zeros<arma::mat>(N_R, M);
	arma::mat mean_X;


	for (unsigned int m = 0 ; m < M; ++m ) {

		mean_X = Lambda[m] * Ybar;
		mean_X.each_col() += Mu.col(m);


		log_Nx.col(m) = log_gaussian_pdf_vec_X(Xbar, mean_X, Psi[m]);
		log_Ny.col(m) = log_gaussian_pdf_vec_general(Ybar, Nu.col(m), Sigma[m]);

	}

	arma::mat normalizers = arma::mat(2, N_R);

	normalizers.row(0) = arma::max(log_Nx, 1).t();
	normalizers.row(1) = arma::max(log_Ny, 1).t();

	arma::vec normalizer = arma::min(normalizers, 0).t();

	arma::vec partial_sum = arma::zeros<arma::vec>(N_R);

	for (unsigned int m = 0 ; m < M; ++m ) {
		partial_sum += arma::exp(std::log(omega(m)) + log_Nx.col(m) + log_Ny.col(m) - normalizer);
	}

	arma::vec denom = arma::log(partial_sum) + normalizer;

	#pragma omp parallel for
	for (unsigned int m = 0 ; m < M; ++m ) {
		Gamma.row(m) = arma::exp(std::log(omega(m)) + log_Nx.col(m) + log_Ny.col(m) - denom).t();
	}


}


arma::vec Nu_update(unsigned int m,
                    arma::mat & Ybar,
                    arma::mat & Gamma) {

	arma::mat test = (Ybar *  Gamma.row(m).t());
	std::cout << arma::dot(Ybar.row(0),Gamma.row(m).t());
	std::cout << test.col(0) << std::endl;

	return (Ybar *  Gamma.row(m).t()) / arma::sum(Gamma.row(m));

}

arma::vec Mu_update(unsigned int m,
                    arma::mat & Xbar,
                    arma::mat & Ybar,
                    std::vector<arma::mat> & Lambda,
                    arma::mat & Gamma
                   ) {

	return ((Xbar - Lambda[m] * Ybar) * Gamma.row(m).t()) / arma::sum(Gamma.row(m));

}

arma::mat Psi_update(unsigned int m,
                     arma::mat & Xbar,
                     arma::mat & Ybar,
                     std::vector<arma::mat> & Lambda,
                     arma::mat & Mu,
                     arma::mat & Gamma) {

	arma::mat::fixed<175, 175> psi_f;
	psi_f.fill(0);


	for (unsigned int i = 0; i < Ybar.n_cols; ++i) {

		psi_f += Gamma.row(m)(i) * (
		             Xbar.col(i) - Lambda[m] * Ybar.col(i) - Mu.col(m)) * (
		             Xbar.col(i) - Lambda[m] * Ybar.col(i) - Mu.col(m)).t();

	}


	return arma::diagmat(psi_f) / arma::sum(Gamma.row(m));

}

arma::mat Lambda_update(unsigned int m,
                        arma::mat & Xbar,
                        arma::mat & Ybar,
                        arma::mat & Mu,
                        arma::mat & Gamma) {

	arma::mat LHS = arma::zeros<arma::mat>(175, 8);
	arma::mat RHS = arma::zeros<arma::mat>(8, 8);

	for (unsigned int i = 0; i < Ybar.n_cols; ++i) {


		LHS += Gamma.row(m)(i) * (Xbar.col(i) - Mu.col(m)) * (Ybar.col(i)).t();
		RHS += Gamma.row(m)(i) * Ybar.col(i) *  (Ybar.col(i)).t();
	}

	return LHS * arma::inv(RHS);

}




void save_model(std::string dir,
                arma::vec & omega,
                arma::mat & Nu,
                std::vector<arma::mat> & Sigma ,
                std::vector<arma::mat> & Lambda,
                arma::mat & Mu,
                std::vector<arma::mat> & Psi) {

	omega.save(dir + "omega.mat", arma::raw_ascii);
	Nu.save(dir + "Nu.mat", arma::raw_ascii);
	Mu.save(dir + "Mu.mat", arma::raw_ascii);

	unsigned int M = omega.n_rows;

	for (unsigned int m = 0; m < M; ++m) {
		Sigma[m].save(dir + "Sigma_" + std::to_string(m) + ".mat", arma::raw_ascii);
		Lambda[m].save(dir + "Lambda_" + std::to_string(m) + ".mat", arma::raw_ascii);
		Psi[m].save(dir + "Psi_" + std::to_string(m) + ".mat", arma::raw_ascii);
	}


}


void load_model(std::string dir,
                arma::vec & omega,
                arma::mat & Nu,
                std::vector<arma::mat> & Sigma ,
                std::vector<arma::mat> & Lambda,
                arma::mat & Mu,
                std::vector<arma::mat> & Psi) {

	omega.load(dir + "omega.mat");
	Nu.load(dir + "Nu.mat");
	Mu.load(dir + "Mu.mat");

	unsigned int M = omega.n_rows;


	for (unsigned int m = 0; m < M; ++m) {

		Sigma.push_back(arma::mat(8, 8));
		Lambda.push_back(arma::mat(175, 8));
		Psi.push_back(arma::mat(175, 175));

		Sigma[Sigma.size() - 1].load(dir + "Sigma_" + std::to_string(m) + ".mat");
		Lambda[m].load(dir + "Lambda_" + std::to_string(m) + ".mat");
		Psi[m].load(dir + "Psi_" + std::to_string(m) + ".mat");
	}


}







arma::mat XQ_to_YQ_GM( arma::mat & X_Q,
                       arma::vec & omega,
                       arma::mat & Nu,
                       std::vector<arma::mat> & Sigma ,
                       std::vector<arma::mat> & Lambda,
                       arma::mat & Mu,
                       std::vector<arma::mat> & Psi) {

	unsigned int N_Q = X_Q.n_cols;
	arma::mat Y_Q = arma::mat(8, N_Q);

	arma::vec log_two_pi_det = arma::vec(N_Q);
	std::vector<arma::mat> cov_inv;

	for (unsigned int m = 0; m < omega.n_rows; ++m) {

		arma::mat cov = Psi[m] + Lambda[m] * Sigma[m] * Lambda[m].t();
		arma::vec eigval = arma::eig_sym( 2 * arma::datum::pi * cov ) ;
		log_two_pi_det(m) = arma::sum(arma::log(eigval));
		cov_inv.push_back(arma::inv(cov));

	}

	boost::progress_display show_progress(N_Q);

	#pragma omp parallel for
	for (unsigned int i = 0; i < N_Q; ++i) {

		Y_Q.col(i) = infer_y_from_x(X_Q.col(i),
		                            omega,
		                            Nu,
		                            Sigma,
		                            Lambda,
		                            Mu,
		                            Psi,
		                            log_two_pi_det,
		                            cov_inv);
		++show_progress;

	}

	return Y_Q;


}

arma::vec infer_y_from_x(arma::vec Xq,
                         arma::vec & omega,
                         arma::mat & Nu,
                         std::vector<arma::mat> & Sigma ,
                         std::vector<arma::mat> & Lambda,
                         arma::mat & Mu,
                         std::vector<arma::mat> & Psi,
                         arma::vec & log_two_pi_det,
                         std::vector<arma::mat> & cov_inv
                        ) {

	arma::mat mu_y_given_m_x = arma::mat(8, omega.n_rows);
	arma::vec X = arma::vec(175);
	arma::vec log_p_x_given_m = arma::vec(omega.n_rows);

	for (unsigned int m = 0; m < omega.n_rows; ++m) {


		X = Xq - Mu.col(m) - Lambda[m] * Nu.col(m);
		log_p_x_given_m(m) = - 0.5 * log_two_pi_det(m) - 0.5 * arma::dot(X, cov_inv[m] * X);

		mu_y_given_m_x.col(m) = (Nu.col(m) + arma::inv(arma::inv(Sigma[m]) +
		                         Lambda[m].t() * arma::diagmat(1. / Psi[m].diag()) * Lambda[m]) * Lambda[m].t() * arma::diagmat(1. / Psi[m].diag()) *
		                         (X));
	}

	arma::vec log_p_m_given_x = arma::log(omega) + log_p_x_given_m;
	double normalizer = log_p_m_given_x.max();

	log_p_m_given_x = log_p_m_given_x - ( std::log(arma::sum(arma::exp(log_p_m_given_x - normalizer))) + normalizer);

	return mu_y_given_m_x * arma::exp(log_p_m_given_x);


}








arma::mat Sigma_update(unsigned int m,
                       arma::mat & Ybar,
                       arma::mat & Nu,
                       arma::mat & Gamma) {

	arma::mat::fixed<8, 8> sigma_f;
	sigma_f.fill(0);

	for (unsigned int i = 0; i < Ybar.n_cols; ++i) {

		sigma_f += Gamma.row(m)(i) * (Ybar.col(i) - Nu.col(m)) * (Ybar.col(i) - Nu.col(m)).t();

	}

	return sigma_f;

}


double bic_score(unsigned int N_R, double icll, unsigned M) {

	double k = (M - 1) + M * 8 + M * 8. * (8 + 1) / 2. + M * 175 + M * 175 + M * 175 * 8;

	return k * std::log(N_R) - 2 * icll;

}
