#include <RcppEigen.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppEigen)]]

// Function to perform the calculations
// [[Rcpp::export]]
Eigen::Array<bool, Eigen::Dynamic, 1> createBoolean(Eigen::VectorXd v) {
  return v.array() >= 0;
}
// [[Rcpp::export]]
Eigen::Array<bool, Eigen::Dynamic, 1> createBoolean2(Eigen::VectorXd v) {
  return v.array() > 0;
}


// [[Rcpp::export]]
Eigen::Array<bool, Eigen::Dynamic, 1> logicalAnd(Eigen::Array<bool, Eigen::Dynamic, 1> vec1,
                                                 Eigen::Array<bool, Eigen::Dynamic, 1> vec2) {
  return (vec1.cast<int>() + vec2.cast<int>() == 2).cast<bool>();
}

// [[Rcpp::export]]
Eigen::VectorXd SubsetVector(Eigen::VectorXd v, Eigen::Array<bool, Eigen::Dynamic, 1> boolVec) {
  return v.array().select(v.array(), Eigen::VectorXd::Zero(v.size())) * boolVec.cast<double>();
}


// [[Rcpp::export]]
double sumVector(Eigen::VectorXd v) {
  return v.sum();
}

// [[Rcpp::export]]
Eigen::VectorXd vcols(Eigen::MatrixXd vab, int idx) {
  return vab.col(idx);
}


// [[Rcpp::export]]
Eigen::ArrayXd elementWiseMultiplication(const Eigen::ArrayXd& array1, const Eigen::ArrayXd& array2) {
  return array1 * array2;
}

// [[Rcpp::export]]
NumericVector elementWiseMultiply(NumericVector x, NumericVector y) {
  int n = x.size();
  NumericVector result(n);
  for (int i = 0; i < n; ++i) {
    result[i] = x[i] * y[i];
  }
  return result;
}

// [[Rcpp::export]]
Eigen::VectorXd Update_Eigen(Eigen::MatrixXd vab, Eigen::VectorXd gab, Eigen::VectorXd gabt, int n) {
  int div = 200;
  Eigen::VectorXd vab1 = vcols(vab,0);
  Eigen::VectorXd vab2 = vcols(vab,1);
  for(int j = 0; j < 100; j++) {
    Eigen::VectorXd vs(n);

    if(j % 3 == 0) {
      vs = vab1 + vab2;
    } else if(j % 3 == 1) {
      vs = vab1.array()*vab1.array() + vab2.array()*vab2.array();
    } else {
      vs = vab1.array()*vab1.array() - vab2.array()*vab2.array();
    }
    double vsmax = vs.maxCoeff();
    double vsmin = vs.minCoeff();
    double vsint = (vsmax - vsmin) / div;
    double ostotal = 0.0;

    for(int i = 0; i < div; i++) {
      Eigen::Array<bool, Eigen::Dynamic, 1> id = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(n, false);
      if(i < (div-1)){
        id = logicalAnd(createBoolean(vs.array() - vsmin - i * vsint),createBoolean2(vsmin + (i+1) * vsint - vs.array()));
      }else{
        id = logicalAnd(createBoolean(vs.array() - vsmin - i * vsint),createBoolean(vsmax - vs.array()));
      }

      Eigen::VectorXd subgabt = SubsetVector(gabt,id);
      Eigen::VectorXd subgab = SubsetVector(gab,id);
      double gs = sumVector(subgabt);
      double g0s = sumVector(subgab);

      if(g0s > 0) {
        ostotal += gs;
        gab = (id).select(gab/g0s*gs,gab);
      } else {
        gab = (id).select(0,gab);
      }
    }

    gab = gab / ostotal;
  }

  return gab;
}



