///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2012  Whit Armstrong                                    //
//                                                                       //
// This program is free software: you can redistribute it and/or modify  //
// it under the terms of the GNU General Public License as published by  //
// the Free Software Foundation, either version 3 of the License, or     //
// (at your option) any later version.                                   //
//                                                                       //
// This program is distributed in the hope that it will be useful,       //
// but WITHOUT ANY WARRANTY; without even the implied warranty of        //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         //
// GNU General Public License for more details.                          //
//                                                                       //
// You should have received a copy of the GNU General Public License     //
// along with this program.  If not, see <http://www.gnu.org/licenses/>. //

#include <iostream>
#include <vector>
#include <RcppArmadillo.h>
#define NDEBUG
//#include <armadillo>
#include <boost/random.hpp>
#include <cppbugs/cppbugs.hpp>
#include <cppbugs/mcmc.multivariate.normal.hpp>

using namespace arma;
using namespace cppbugs;
//using std::cout;
//using std::endl;

extern "C" SEXP multivariate_sv(SEXP X_, SEXP iterations_, SEXP burn_, SEXP adapt_, SEXP thin_);

uvec lower_diag(const size_t n) {
  uvec ans(n*(n-1)/2);
  size_t idx(0);
  for(size_t i = 1; i < n; i++) {
    for(size_t j = 0; j < i; j++) {
      ans[idx++] = i + j * n;
    }
  }
  return ans;
}


SEXP multivariate_sv(SEXP X_, SEXP iterations_, SEXP burn_, SEXP adapt_, SEXP thin_) {

  const mat X = Rcpp::as<arma::mat>(X_);
  const int iterations = Rcpp::as<int>(iterations_);
  const int burn = Rcpp::as<int>(burn_);
  const int adapt = Rcpp::as<int>(adapt_);
  const int thin = Rcpp::as<int>(thin_);
  const int NR = X.n_rows;
  const int NC = X.n_cols;
  const int OD = NC*(NC-1)/2;
  
  rowvec X_mu = zeros<rowvec>(NC);

  // element indices for lower diagonal
  uvec ld_elems = lower_diag(NC);

  // time series histories of dt and pt
  mat log_dt = randn<mat>(NR,NC);
  mat log_dt_lag(NR,NC);
  rowvec log_dt0 =randn<rowvec>(NC);
  //rowvec dt_tau(NC);

  mat pt = randn<mat>(NR,OD);
  mat pt_lag(NR,OD);
  rowvec pt0 = randn<rowvec>(OD);
  //rowvec pt_tau(OD);

  std::vector<mat> LL_t;
  std::vector<mat> sigma_t;
  for(int i = 0; i < NR; i++) {
    LL_t.push_back(zeros<mat>(NC,NC));
    sigma_t.push_back(zeros<mat>(NC,NC));
  }

  std::function<void ()> model = [&]() {
    // lag of dt -- to guarantee > 0 diagonal of LL
    log_dt_lag.row(0) = log_dt0;
    log_dt_lag.rows(1,log_dt_lag.n_rows-1) = log_dt.rows(0,log_dt.n_rows-2);

    // lag of pt
    pt_lag.row(0) = pt0;
    pt_lag.rows(1,pt_lag.n_rows-1) = pt.rows(0,pt.n_rows-2);

    for(size_t i = 0; i < NR; i++) {
      LL_t[i].diag() = exp(log_dt.row(i));
      //LL_t[i].diag(-1) = pt[i];
      LL_t[i].elem(ld_elems) = pt.row(i);
      sigma_t[i] = LL_t[i] * trans(LL_t[i]);
    }
  };


  MCModel<boost::minstd_rand> m(model);

  // diag of LL
  m.track<Normal>(log_dt0).dnorm(0, 0.0001);
  m.track<Normal>(log_dt).dnorm(log_dt_lag, 1);

  // offdiag of LL
  m.track<Normal>(pt0).dnorm(0, 0.0001);
  m.track<Normal>(pt).dnorm(pt_lag, 1);

  for(int i = 0; i < NR; i++) {
    m.track<ObservedMultivariateNormal>(X.row(i)).dmvnorm(X_mu,sigma_t[i]);
  }

  m.sample(iterations, burn, adapt, thin);

  return Rcpp::List::create(Rcpp::Named("log_dt", m.getNode(log_dt).mean()),
                            Rcpp::Named("pt", m.getNode(pt).mean()),
                            Rcpp::Named("ar", m.acceptance_ratio()));

}
