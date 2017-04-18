#ifndef PARTB_HPP
#define PARTB_HPP

#include <armadillo>
#include <boost/progress.hpp>

void init(unsigned int M,
          arma::mat & Xbar,
          arma::mat & Ybar,
          arma::vec & omega,
          arma::mat & Nu,
          std::vector<arma::mat> & Sigma ,
          std::vector<arma::mat> & Lambda,
          arma::mat & Mu,
          std::vector<arma::mat> & Psi,
          arma::mat & Gamma);

double ICLL(arma::mat & Xbar,
            arma::mat & Ybar,
            arma::vec & omega,
            arma::mat & Nu,
            std::vector<arma::mat> & Sigma ,
            std::vector<arma::mat> & Lambda,
            arma::mat & Mu,
            std::vector<arma::mat> & Psi);

void M_step(arma::mat & Xbar,
            arma::mat & Ybar,
            arma::vec & omega,
            arma::mat & Nu,
            std::vector<arma::mat> & Sigma ,
            std::vector<arma::mat> & Lambda,
            arma::mat & Mu,
            std::vector<arma::mat> & Psi,
            arma::mat & Gamma);

void E_step(arma::mat & Xbar,
            arma::mat & Ybar,
            arma::vec & omega,
            arma::mat & Nu,
            std::vector<arma::mat> & Sigma ,
            std::vector<arma::mat> & Lambda,
            arma::mat & Mu,
            std::vector<arma::mat> & Psi,
            arma::mat & Gamma);


arma::vec Nu_update(unsigned int m,
                    arma::mat & Ybar,
                    arma::mat & Gamma);

arma::vec Mu_update(unsigned int m,
                    arma::mat & Xbar,
                    arma::mat & Ybar,
                    std::vector<arma::mat> & Lambda,
                    arma::mat & Gamma
                   );

arma::mat Psi_update(unsigned int m,
                     arma::mat & Xbar,
                     arma::mat & Ybar,
                     std::vector<arma::mat> & Lambda,
                     arma::mat & Mu,
                     arma::mat & Gamma);

arma::mat Lambda_update(unsigned int m,
                        arma::mat & Xbar,
                        arma::mat & Ybar,
                        arma::mat & Mu,
                        arma::mat & Gamma);

arma::mat Sigma_update(unsigned int m,
                       arma::mat & Ybar,
                       arma::mat & Nu,
                       arma::mat & Gamma) ;


arma::vec log_gaussian_pdf_vec_X(arma::mat & Xbar,
                                 arma::mat & mean,
                                 arma::mat & cov);



arma::vec log_gaussian_pdf_vec_general(arma::mat & Ybar,
                                       arma::vec  mean,
                                       arma::mat & cov) ;

double bic_score(unsigned int N_R, double icll, unsigned M);

arma::mat XQ_to_YQ_GM( arma::mat & X_Q,
                       arma::vec & omega,
                       arma::mat & Nu,
                       std::vector<arma::mat> & Sigma ,
                       std::vector<arma::mat> & Lambda,
                       arma::mat & Mu,
                       std::vector<arma::mat> & Psi);

arma::vec infer_y_from_x(arma::vec Xq,
                         arma::vec & omega,
                         arma::mat & Nu,
                         std::vector<arma::mat> & Sigma ,
                         std::vector<arma::mat> & Lambda,
                         arma::mat & Mu,
                         std::vector<arma::mat> & Psi,
                         arma::vec & log_two_pi_det,
                         std::vector<arma::mat> & cov_inv
                        );

void load_model(std::string dir,
                arma::vec & omega,
                arma::mat & Nu,
                std::vector<arma::mat> & Sigma ,
                std::vector<arma::mat> & Lambda,
                arma::mat & Mu,
                std::vector<arma::mat> & Psi) ;

void save_model(std::string dir,
                arma::vec & omega,
                arma::mat & Nu,
                std::vector<arma::mat> & Sigma ,
                std::vector<arma::mat> & Lambda,
                arma::mat & Mu,
                std::vector<arma::mat> & Psi);

#endif