/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// Copyright 2013, 2016 Schmidt

#include <gsl/gsl_sf_bessel.h>
#include <string.h>
#include <math.h>

#include "base/linalg/linalg.h"
#include "base/utils/utils.h"

// R.h and Rinternals.h needs to be included after Rconfig.h
#include "pbdBASE.h"

double univariate_matern_spatial(double *PARAM, double *l1, double *l2)
{
  double cov_val = 0.0;
  double expr = 0.0;
  double con = 0.0;
  double sigma_square = PARAM[0], range = PARAM[1], smoothness = PARAM[2];

  con = pow(2, (smoothness - 1)) * tgamma(smoothness);
  con = 1.0/con;
  con = sigma_square * con;

  expr = sqrt(pow(l1[0] - l2[0], 2) + pow(l1[1] - l2[1], 2)) / range;

  if(expr == 0){
    cov_val = sigma_square;
  }else{
    cov_val = con * pow(expr, smoothness) * gsl_sf_bessel_Knu(smoothness, expr);
  }
  //printf("loc1: %f, %f, %f; loc2: %f, %f, %f; dist: %f; cov: %f \n", l1[0], l1[1], l1[2], l2[0], l2[1], l2[2], expr, cov_val);
  return cov_val;
}

double bivariate_matern_parsimonious_spatial(double *PARAM, double *l1, double *l2)
{
  double cov_val = 0.0;
  double expr = 0.0;
  double con = 0.0;
  double sigma_square = 0.0, range = PARAM[2], smoothness = 0.0;

  if(l1[3] == l2[3]){
    if(l1[3] == 1){
      sigma_square = PARAM[0];
      smoothness = PARAM[3];
    }else{
      sigma_square = PARAM[1];
      smoothness = PARAM[4];
    }
  }else{
    sigma_square = PARAM[5] * sqrt(PARAM[0] * PARAM[1]);
    smoothness = 0.5 * (PARAM[3] + PARAM[4]);
  }

  con = pow(2, (smoothness - 1)) * tgamma(smoothness);
  con = 1.0/con;
  con = sigma_square * con;

  expr = sqrt(pow(l1[0] - l2[0], 2) + pow(l1[1] - l2[1], 2)) / range;

  if(expr == 0){
    cov_val = sigma_square;
  }else{
    cov_val = con * pow(expr, smoothness) * gsl_sf_bessel_Knu(smoothness, expr);
  }
  return cov_val;
}

double univariate_matern_schlather_spacetime(double *PARAM, double *l1, double *l2)
{
  double cov_val = 0.0;
  double expr = 0.0;
  double con = 0.0, con_new = 0.0;
  double sigma_square = PARAM[0], range = PARAM[1], smoothness = PARAM[2];
  double vel_mean_x = PARAM[3], vel_mean_y = PARAM[4];
  double vel_variance_chol_11 = PARAM[5], vel_variance_chol_12 = PARAM[6], vel_variance_chol_22 = PARAM[7];
  
  double l1x_new, l1y_new, l2x_new, l2y_new, xlag, ylag, tlag;
  double vel_variance11, vel_variance22, vel_variance12;
  double vel_variance11_new, vel_variance22_new, vel_variance12_new;
  double Inv_11, Inv_22, Inv_12, det;

  con = pow(2,(smoothness - 1)) * tgamma(smoothness);
  con = 1.0/con;

  l1x_new = l1[0] - vel_mean_x * l1[2];
  l1y_new = l1[1] - vel_mean_y * l1[2];
  l2x_new = l2[0] - vel_mean_x * l2[2];
  l2y_new = l2[1] - vel_mean_y * l2[2];

  xlag = l1x_new - l2x_new;
  ylag = l1y_new - l2y_new;
  tlag = l1[2] - l2[2];

  vel_variance11 = pow(vel_variance_chol_11, 2);
  vel_variance22 = pow(vel_variance_chol_12, 2) + pow(vel_variance_chol_22, 2);
  vel_variance12 = vel_variance_chol_11 * vel_variance_chol_12;

  vel_variance11_new = 1 + vel_variance11 * pow(tlag, 2);
  vel_variance22_new = 1 + vel_variance22 * pow(tlag, 2);
  vel_variance12_new = vel_variance12 * pow(tlag, 2);

  det = vel_variance11_new * vel_variance22_new - pow(vel_variance12_new, 2);
  //printf("location1: %f, %f, %f; location2: %f, %f, %f \n", l1[0], l1[1], l1[2], l2[0], l2[1], l2[2]);

  Inv_11 = vel_variance22_new / det;
  Inv_22 = vel_variance11_new / det;
  Inv_12 = -vel_variance12_new / det;
 
  expr = sqrt(pow(xlag, 2) * Inv_11 + 2 * xlag * ylag * Inv_12 + pow(ylag, 2) * Inv_22 ) / range;

  con_new = sigma_square * con / sqrt(det);

  if(expr == 0){
    cov_val = sigma_square / sqrt(det);
  }else{
    cov_val = con_new * pow(expr, smoothness) * gsl_sf_bessel_Knu(smoothness, expr);
  }
  return cov_val;
}

double bivariate_matern_salvana_single_advection_spacetime(double *PARAM, double *l1, double *l2)
{
  double cov_val = 0.0;
  double expr = 0.0;
  double con = 0.0, con_new = 0.0;
  double sigma_square = 0.0, range = PARAM[2], smoothness = 0.0;

  if(l1[3] == l2[3]){
    if(l1[3] == 1){
      sigma_square = PARAM[0];
      smoothness = PARAM[3];
    }else{
      sigma_square = PARAM[1];
      smoothness = PARAM[4];
    }
  }else{
    sigma_square = PARAM[5] * sqrt(PARAM[0] * PARAM[1]);
    smoothness = 0.5 * (PARAM[3] + PARAM[4]);
  }

  double vel_mean_x = PARAM[6], vel_mean_y = PARAM[7];
  double vel_variance_chol_11 = PARAM[8], vel_variance_chol_12 = PARAM[9], vel_variance_chol_22 = PARAM[10];
  
  double l1x_new, l1y_new, l2x_new, l2y_new, xlag, ylag, tlag;
  double vel_variance11, vel_variance22, vel_variance12;
  double vel_variance11_new, vel_variance22_new, vel_variance12_new;
  double Inv_11, Inv_22, Inv_12, det;

  con = pow(2,(smoothness - 1)) * tgamma(smoothness);
  con = 1.0/con;

  l1x_new = l1[0] - vel_mean_x * l1[2];
  l1y_new = l1[1] - vel_mean_y * l1[2];
  l2x_new = l2[0] - vel_mean_x * l2[2];
  l2y_new = l2[1] - vel_mean_y * l2[2];

  xlag = l1x_new - l2x_new;
  ylag = l1y_new - l2y_new;
  tlag = l1[2] - l2[2];

  vel_variance11 = pow(vel_variance_chol_11, 2);
  vel_variance22 = pow(vel_variance_chol_12, 2) + pow(vel_variance_chol_22, 2);
  vel_variance12 = vel_variance_chol_11 * vel_variance_chol_12;

  vel_variance11_new = 1 + vel_variance11 * pow(tlag, 2);
  vel_variance22_new = 1 + vel_variance22 * pow(tlag, 2);
  vel_variance12_new = vel_variance12 * pow(tlag, 2);

  det = vel_variance11_new * vel_variance22_new - pow(vel_variance12_new, 2);
  //printf("location1: %f, %f, %f; location2: %f, %f, %f \n", l1[0], l1[1], l1[2], l2[0], l2[1], l2[2]);

  Inv_11 = vel_variance22_new / det;
  Inv_22 = vel_variance11_new / det;
  Inv_12 = -vel_variance12_new / det;
 
  expr = sqrt(pow(xlag, 2) * Inv_11 + 2 * xlag * ylag * Inv_12 + pow(ylag, 2) * Inv_22 ) / range;

  con_new = sigma_square * con / sqrt(det);

  if(expr == 0){
    cov_val = sigma_square / sqrt(det);
  }else{
    cov_val = con_new * pow(expr, smoothness) * gsl_sf_bessel_Knu(smoothness, expr);
  }
  return cov_val;
}

void covfunc_(int *MODEL_NUM, double *PARAM_VECTOR, double *L1, double *L2, double *gi)
{

  if(*MODEL_NUM == 1){
    *gi = univariate_matern_spatial(PARAM_VECTOR, L1, L2);
  }else if(*MODEL_NUM == 2){
    *gi = univariate_matern_schlather_spacetime(PARAM_VECTOR, L1, L2);
  }else if(*MODEL_NUM == 3){
    *gi = bivariate_matern_parsimonious_spatial(PARAM_VECTOR, L1, L2);
  }else if(*MODEL_NUM == 4){
    *gi = bivariate_matern_salvana_single_advection_spacetime(PARAM_VECTOR, L1, L2);
  }
}

SEXP R_MKSUBMAT(SEXP GBLX, SEXP LDIM, SEXP DESCX)
{
  SEXP SUBX;
  PROTECT(SUBX = allocMatrix(REALSXP, INTEGER(LDIM)[0], INTEGER(LDIM)[1]));
  
  mksubmat_(REAL(GBLX), REAL(SUBX), INTEGER(DESCX));
  
  UNPROTECT(1);
  return SUBX;
} 

SEXP R_COVSUBMAT(SEXP MODEL, SEXP PARAM, SEXP GBLX, SEXP LDIM, SEXP DESCX)
{
  SEXP SUBX;
  PROTECT(SUBX = allocMatrix(REALSXP, INTEGER(LDIM)[0], INTEGER(LDIM)[1]));
  
  covsubmat_(INTEGER(MODEL), REAL(PARAM), REAL(GBLX), REAL(SUBX), INTEGER(DESCX));

  UNPROTECT(1);
  return SUBX;
} 


SEXP R_CROSSCOVSUBMAT(SEXP MODEL, SEXP PARAM, SEXP GBLX, SEXP LDIM, SEXP DESCX)
{
  SEXP SUBX;
  //PROTECT(SUBX = allocMatrix(REALSXP, 2 * INTEGER(LDIM)[0], 2 * INTEGER(LDIM)[1]));
  PROTECT(SUBX = allocMatrix(REALSXP, INTEGER(LDIM)[0], INTEGER(LDIM)[1]));
  
  crosscovsubmat_(INTEGER(MODEL), REAL(PARAM), REAL(GBLX), REAL(SUBX), INTEGER(DESCX));

  UNPROTECT(1);
  return SUBX;
} 



SEXP R_MKGBLMAT(SEXP SUBX, SEXP DESCX, SEXP RDEST, SEXP CDEST)
{
  SEXP GBLX;
  PROTECT(GBLX = allocMatrix(REALSXP, INTEGER(DESCX)[2], INTEGER(DESCX)[3]));
  
  mkgblmat_(REAL(GBLX), REAL(SUBX), INTEGER(DESCX), INTEGER(RDEST), 
    INTEGER(CDEST));
  
  UNPROTECT(1);
  return GBLX;
} 



SEXP R_DALLREDUCE(SEXP X, SEXP LDIM, SEXP DESCX, SEXP OP, SEXP SCOPE)
{
  const int m = INTEGER(DESCX)[2];
  const int n = INTEGER(DESCX)[3];
  
  SEXP CPX;
  PROTECT(CPX = allocMatrix(REALSXP, INTEGER(LDIM)[0], INTEGER(LDIM)[1]));
  
  memcpy(REAL(CPX), REAL(X), m*n*sizeof(double));

#ifdef FC_LEN_T
  dallreduce_(REAL(CPX), INTEGER(DESCX), CHARPT(OP, 0), CHARPT(SCOPE, 0), (FC_LEN_T) strlen(CHARPT(OP, 0)), (FC_LEN_T) strlen(CHARPT(SCOPE, 0)));
#else
  dallreduce_(REAL(CPX), INTEGER(DESCX), CHARPT(OP, 0), CHARPT(SCOPE, 0));
#endif
  
  UNPROTECT(1);
  return CPX;
} 



SEXP R_PTRI2ZERO(SEXP UPLO, SEXP DIAG, SEXP X, SEXP LDIM, SEXP DESCX)
{
  const int m = INTEGER(LDIM)[0];
  const int n = INTEGER(LDIM)[1];
  
  SEXP CPX;
  PROTECT(CPX = allocMatrix(REALSXP, m, n));
  
  memcpy(REAL(CPX), REAL(X), m*n*sizeof(double));
  
#ifdef FC_LEN_T
  ptri2zero_(CHARPT(UPLO, 0), CHARPT(DIAG, 0), REAL(CPX), INTEGER(DESCX), (FC_LEN_T) strlen(CHARPT(UPLO, 0)), (FC_LEN_T) strlen(CHARPT(DIAG, 0)));
#else
  ptri2zero_(CHARPT(UPLO, 0), CHARPT(DIAG, 0), REAL(CPX), INTEGER(DESCX));
#endif
  
  UNPROTECT(1);
  return CPX;
}



SEXP R_PDSWEEP(SEXP X, SEXP LDIM, SEXP DESCX, SEXP VEC, SEXP LVEC, SEXP MARGIN, SEXP FUN)
{
  const int m = INTEGER(LDIM)[0];
  const int n = INTEGER(LDIM)[1];
  int IJ = 1;
  
  SEXP CPX;
  PROTECT(CPX = allocMatrix(REALSXP, m, n));
  
  memcpy(REAL(CPX), REAL(X), m*n*sizeof(double));
  
  pdsweep(REAL(CPX), IJ, IJ, INTEGER(DESCX), REAL(VEC), INTEGER(LVEC)[0], INTEGER(MARGIN)[0], CHARPT(FUN, 0)[0]);
  
  UNPROTECT(1);
  return CPX;
}



SEXP R_PDGDGTK(SEXP X, SEXP LDIM, SEXP DESCX, SEXP LDIAG, SEXP RDEST, SEXP CDEST)
{
  UNUSED(LDIM);
  int IJ = 1;
  
  SEXP DIAG;
  PROTECT(DIAG = allocVector(REALSXP, INTEGER(LDIAG)[0]));
  
  pdgdgtk_(REAL(X), &IJ, &IJ, INTEGER(DESCX), REAL(DIAG), 
    INTEGER(RDEST), INTEGER(CDEST));
  
  UNPROTECT(1);
  return DIAG;
}



SEXP R_PDDIAGMK(SEXP LDIM, SEXP DESCX, SEXP DIAG, SEXP LDIAG)
{
  const int m = INTEGER(LDIM)[0];
  const int n = INTEGER(LDIM)[1];
  int IJ = 1;
  
  SEXP X;
  PROTECT(X = allocMatrix(REALSXP, m, n));
  
  pddiagmk_(REAL(X), &IJ, &IJ, INTEGER(DESCX), REAL(DIAG), INTEGER(LDIAG));
  
  UNPROTECT(1);
  return X;
}



SEXP R_DHILBMK(SEXP N)
{
  int n = INTEGER(N)[0];
  
  SEXP X;
  PROTECT(X = allocMatrix(REALSXP, n, n));
  
  dhilbmk_(&n, REAL(X));
  
  UNPROTECT(1);
  return X;
}



SEXP R_PDHILBMK(SEXP LDIM, SEXP DESCX)
{
  SEXP X;
  PROTECT(X = allocMatrix(REALSXP, INTEGER(LDIM)[0], INTEGER(LDIM)[1]));
  
  pdhilbmk_(REAL(X), INTEGER(DESCX));
  
  UNPROTECT(1);
  return X;
}



SEXP R_PDMKCPN1(SEXP LDIM, SEXP DESCX, SEXP COEF)
{
  const int m = INTEGER(LDIM)[0];
  const int n = INTEGER(LDIM)[1];
  
  SEXP X;
  PROTECT(X = allocMatrix(REALSXP, m, n));
  
  pdmkcpn1_(REAL(X), INTEGER(DESCX), REAL(COEF));
  
  UNPROTECT(1);
  return X;
}
