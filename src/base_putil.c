/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// Copyright 2013, 2016 Schmidt

#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include <string.h>
#include <math.h>

#include "base/linalg/linalg.h"
#include "base/utils/utils.h"

// R.h and Rinternals.h needs to be included after Rconfig.h
#include "pbdBASE.h"

void matrix_print(const gsl_matrix * M)
{
  // Get the dimension of the matrix.
  int rows = M->size1;
  int cols = M->size2;

  // Now print out the data in a square format.
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      printf(" %f ", gsl_matrix_get(M, i, j));
    }
    printf("\n");
  } 
}

gsl_matrix *invert_a_matrix(gsl_matrix *matrix, int size)
{
  gsl_permutation *p = gsl_permutation_alloc(size);
  int s;

  // Compute the LU decomposition of this matrix
  gsl_linalg_LU_decomp(matrix, p, &s);

  // Compute the  inverse of the LU decomposition
  gsl_matrix *inv = gsl_matrix_alloc(size, size);
  gsl_linalg_LU_invert(matrix, p, inv);

  gsl_permutation_free(p);

  return inv;
}

double matrix_determinant(gsl_matrix *matrix, int size)
{
  gsl_permutation *p = gsl_permutation_alloc(size);
  int s;

  // Compute the LU decomposition of this matrix
  gsl_linalg_LU_decomp(matrix, p, &s);

  // Compute the determinant from the LU decomposition
  double det = gsl_linalg_LU_det(matrix, s);

  gsl_permutation_free(p);

  return det;
}

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
  //printf("location1: %f, %f, %f, %f; location2: %f, %f, %f, %f \n", l1[0], l1[1], l1[2], l1[3], l2[0], l2[1], l2[2], l2[3]);

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
  
  double l1x_new = 0.0, l1y_new = 0.0, l2x_new = 0.0, l2y_new = 0.0, xlag = 0.0, ylag = 0.0, tlag = 0.0;
  double vel_variance11 = 0.0, vel_variance22 = 0.0, vel_variance12 = 0.0;
  double vel_variance11_new = 0.0, vel_variance22_new = 0.0, vel_variance12_new = 0.0;
  double Inv_11 = 0.0, Inv_22 = 0.0, Inv_12 = 0.0, det = 0.0;

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

double bivariate_matern_salvana_multiple_advection_spacetime(double *PARAM, double *l1, double *l2)
{
  double cov_val = 0.0;
  double expr = 0.0, det = 0.0;
  double con = 0.0, con_new = 0.0;
  double sigma_square = 0.0, range = PARAM[2], smoothness = 0.0;
  int i, j, ulag, t1, t2, variable1, variable2;

  t1 = l1[2];
  t2 = l2[2];
  ulag = t1 - t2;
    
  variable1 = l1[3];
  variable2 = l2[3];

  gsl_matrix *vel_mean = gsl_matrix_alloc(4, 1);
  gsl_matrix *vel_variance_chol = gsl_matrix_alloc(4, 4);
  gsl_matrix *vel_variance_temp = gsl_matrix_alloc(4, 4);
  gsl_matrix *hlag = gsl_matrix_alloc(2, 1);

  gsl_matrix *expr_temp1 = gsl_matrix_alloc(1, 2); 
  gsl_matrix *expr_temp2 = gsl_matrix_alloc(1, 1); 

  gsl_matrix *vel_mean_marginal = gsl_matrix_alloc(2, 1);
  gsl_matrix *vel_variance_marginal = gsl_matrix_alloc(2, 2);
  gsl_matrix *vel_variance_new = gsl_matrix_alloc(2, 2);
  gsl_matrix *vel_variance_inverse = gsl_matrix_alloc(2, 2);

  gsl_matrix *vel_mean_marginal1 = gsl_matrix_alloc(2, 1);
  gsl_matrix *vel_mean_marginal2 = gsl_matrix_alloc(2, 1);
  gsl_matrix *vel_variance = gsl_matrix_alloc(4, 4);
  gsl_matrix *vel_variance_copy = gsl_matrix_alloc(4, 4);
  gsl_matrix *T_matrix = gsl_matrix_alloc(2, 4);
  gsl_matrix *vel_mean_cross = gsl_matrix_alloc(2, 1);
  gsl_matrix *T_matrix_squared = gsl_matrix_alloc(4, 4);
  gsl_matrix *vel_variance_inverse_cross = gsl_matrix_alloc(4, 4);    
  gsl_matrix *vel_variance_new_quasi_temp1 = gsl_matrix_alloc(4, 4);    
  gsl_matrix *vel_variance_new_quasi_temp2 = gsl_matrix_alloc(4, 4); 
  gsl_matrix *vel_variance_new_quasi_temp3 = gsl_matrix_alloc(2, 4); 
  gsl_matrix *vel_variance_new_quasi_temp4 = gsl_matrix_alloc(2, 2); 
  gsl_matrix *vel_variance_new_quasi = gsl_matrix_alloc(2, 2);
  gsl_matrix *denom = gsl_matrix_alloc(4, 4); 
  gsl_matrix *denom_temp = gsl_matrix_alloc(4, 4); 


  gsl_matrix_set_zero(vel_variance_chol);

  unsigned int count = 0;
  for(i = 0; i < 4; i++){
    for(j = i; j < 4; j++){
      gsl_matrix_set(vel_variance_chol, i, j, PARAM[10 + count]); 
      count += 1;
    }
  }

  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, vel_variance_chol, vel_variance_chol, 0.0, vel_variance_temp);

  gsl_matrix_set(hlag, 0, 0, l1[0] - l2[0]);
  gsl_matrix_set(hlag, 1, 0, l1[1] - l2[1]);

  if(variable1 == variable2){

    if(variable1 == 1){

      sigma_square = PARAM[0];
      smoothness = PARAM[3];

      //gsl_matrix_view vel_mean_sub = gsl_matrix_submatrix(vel_mean, 0, 0, 2, 1);
      gsl_matrix_view vel_variance_sub = gsl_matrix_submatrix(vel_variance_temp, 0, 0, 2, 2);

      //vel_mean_marginal = &vel_mean_sub.matrix;
      for(j = 0; j < 2; j++){
        gsl_matrix_set(vel_mean_marginal, j, 0, PARAM[6 + j]); 
      }
      vel_variance_marginal = &vel_variance_sub.matrix;

    }else{

      sigma_square = PARAM[1];
      smoothness = PARAM[4];

      //gsl_matrix_view vel_mean_sub = gsl_matrix_submatrix(vel_mean, 2, 0, 2, 1);
      gsl_matrix_view vel_variance_sub = gsl_matrix_submatrix(vel_variance_temp, 2, 2, 2, 2);

      //vel_mean_marginal = &vel_mean_sub.matrix;
      for(j = 0; j < 2; j++){
        gsl_matrix_set(vel_mean_marginal, j, 0, PARAM[8 + j]); 
      }
      vel_variance_marginal = &vel_variance_sub.matrix;

    }

    gsl_matrix_scale(vel_mean_marginal, ulag);
    gsl_matrix_scale(vel_variance_marginal, pow(ulag, 2));

    gsl_matrix_sub(hlag, vel_mean_marginal);

    gsl_matrix_set_identity(vel_variance_new);
    gsl_matrix_add(vel_variance_new, vel_variance_marginal);

    vel_variance_inverse = invert_a_matrix(vel_variance_new, 2);   

    det = matrix_determinant(vel_variance_new, 2);

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, hlag, vel_variance_inverse, 0.0, expr_temp1);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, expr_temp1, hlag, 0.0, expr_temp2);

    expr = sqrt(gsl_matrix_get(expr_temp2, 0, 0)) / range;
 
  }else{

    sigma_square = PARAM[5] * sqrt(PARAM[0] * PARAM[1]);
    smoothness = 0.5 * (PARAM[3] + PARAM[4]);
 
    if(variable1 == 1 & variable2 == 2){

      for(j = 0; j < 2; j++){
        gsl_matrix_set(vel_mean_marginal1, j, 0, PARAM[6 + j]); 
      }
      for(j = 0; j < 2; j++){
        gsl_matrix_set(vel_mean_marginal2, j, 0, PARAM[8 + j]); 
      }

      gsl_matrix_memcpy(vel_variance, vel_variance_temp);

    }else if(variable1 == 2 & variable2 == 1){

      for(j = 0; j < 2; j++){
        gsl_matrix_set(vel_mean_marginal1, j, 0, PARAM[8 + j]); 
      }
      for(j = 0; j < 2; j++){
        gsl_matrix_set(vel_mean_marginal2, j, 0, PARAM[6 + j]); 
      }

      gsl_matrix_memcpy(vel_variance, vel_variance_temp);

      gsl_matrix_set(vel_variance, 0, 0, gsl_matrix_get(vel_variance_temp, 2, 2));
      gsl_matrix_set(vel_variance, 0, 1, gsl_matrix_get(vel_variance_temp, 2, 3));
      gsl_matrix_set(vel_variance, 1, 0, gsl_matrix_get(vel_variance_temp, 3, 2));
      gsl_matrix_set(vel_variance, 1, 1, gsl_matrix_get(vel_variance_temp, 3, 3));

      gsl_matrix_set(vel_variance, 2, 2, gsl_matrix_get(vel_variance_temp, 0, 0));
      gsl_matrix_set(vel_variance, 2, 3, gsl_matrix_get(vel_variance_temp, 0, 1));
      gsl_matrix_set(vel_variance, 3, 2, gsl_matrix_get(vel_variance_temp, 1, 0));
      gsl_matrix_set(vel_variance, 3, 3, gsl_matrix_get(vel_variance_temp, 1, 1));

      gsl_matrix_set(vel_variance, 0, 2, gsl_matrix_get(vel_variance_temp, 2, 0));
      gsl_matrix_set(vel_variance, 0, 3, gsl_matrix_get(vel_variance_temp, 2, 1));
      gsl_matrix_set(vel_variance, 1, 2, gsl_matrix_get(vel_variance_temp, 3, 0));
      gsl_matrix_set(vel_variance, 1, 3, gsl_matrix_get(vel_variance_temp, 3, 1));

      gsl_matrix_set(vel_variance, 2, 0, gsl_matrix_get(vel_variance_temp, 0, 2));
      gsl_matrix_set(vel_variance, 2, 1, gsl_matrix_get(vel_variance_temp, 0, 3));
      gsl_matrix_set(vel_variance, 3, 0, gsl_matrix_get(vel_variance_temp, 1, 2));
      gsl_matrix_set(vel_variance, 3, 1, gsl_matrix_get(vel_variance_temp, 1, 3));
    }


    gsl_matrix_set(vel_mean, 0, 0, gsl_matrix_get(vel_mean_marginal1, 0, 0)); 
    gsl_matrix_set(vel_mean, 1, 0, gsl_matrix_get(vel_mean_marginal1, 1, 0)); 
    gsl_matrix_set(vel_mean, 2, 0, gsl_matrix_get(vel_mean_marginal2, 0, 0)); 
    gsl_matrix_set(vel_mean, 3, 0, gsl_matrix_get(vel_mean_marginal2, 1, 0)); 


    gsl_matrix_set_zero(T_matrix);

    gsl_matrix_set(T_matrix, 0, 0, t1);
    gsl_matrix_set(T_matrix, 1, 1, t1);
    gsl_matrix_set(T_matrix, 0, 2, -t2);
    gsl_matrix_set(T_matrix, 1, 3, -t2);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, T_matrix, vel_mean, 0.0, vel_mean_cross);

    gsl_matrix_sub(hlag, vel_mean_cross);

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, T_matrix, T_matrix, 0.0, T_matrix_squared);

    gsl_matrix_memcpy(vel_variance_copy, vel_variance);
    vel_variance_inverse_cross = invert_a_matrix(vel_variance_copy, 4);   

      //printf("t1: %d; t2: %d; var1: %d; var2: %d \n", t1, t2, variable1, variable2);
      //matrix_print(denom_temp);      

    gsl_matrix_memcpy(vel_variance_new_quasi_temp1, T_matrix_squared);

    gsl_matrix_add(vel_variance_new_quasi_temp1, vel_variance_inverse_cross);

    vel_variance_new_quasi_temp2 = invert_a_matrix(vel_variance_new_quasi_temp1, 4);   

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, T_matrix, vel_variance_new_quasi_temp2, 0.0, vel_variance_new_quasi_temp3);

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, vel_variance_new_quasi_temp3, T_matrix, 0.0, vel_variance_new_quasi_temp4);

    gsl_matrix_set_identity(vel_variance_new_quasi);

    gsl_matrix_sub(vel_variance_new_quasi, vel_variance_new_quasi_temp4);

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, hlag, vel_variance_new_quasi, 0.0, expr_temp1);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, expr_temp1, hlag, 0.0, expr_temp2);

    expr = sqrt(gsl_matrix_get(expr_temp2, 0, 0)) / range;

    gsl_matrix_set_identity(denom);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, vel_variance, T_matrix_squared, 0.0, denom_temp);


    gsl_matrix_add(denom, denom_temp);

    det = matrix_determinant(denom, 4);

  }

  con = pow(2,(smoothness - 1)) * tgamma(smoothness);
  con = 1.0/con;

  con_new = sigma_square * con / sqrt(det);

  if(expr == 0){
    cov_val = sigma_square / sqrt(det);
  }else{
    cov_val = con_new * pow(expr, smoothness) * gsl_sf_bessel_Knu(smoothness, expr);
  }
  return cov_val;

  gsl_matrix_free(vel_mean);
  gsl_matrix_free(vel_variance_chol);
  gsl_matrix_free(vel_variance_temp);
  gsl_matrix_free(hlag);
  gsl_matrix_free(expr_temp1);
  gsl_matrix_free(expr_temp2);

  gsl_matrix_free(vel_mean_marginal);
  gsl_matrix_free(vel_variance_marginal);
  gsl_matrix_free(vel_variance_new);
  gsl_matrix_free(vel_variance_inverse);

  gsl_matrix_free(vel_mean_marginal1);
  gsl_matrix_free(vel_mean_marginal2);
  gsl_matrix_free(vel_variance);
  gsl_matrix_free(vel_variance_copy);
  gsl_matrix_free(T_matrix);
  gsl_matrix_free(vel_mean_cross);
  gsl_matrix_free(T_matrix_squared);
  gsl_matrix_free(vel_variance_inverse_cross);
  gsl_matrix_free(vel_variance_new_quasi_temp1);
  gsl_matrix_free(vel_variance_new_quasi_temp2);
  gsl_matrix_free(vel_variance_new_quasi_temp3);
  gsl_matrix_free(vel_variance_new_quasi_temp4);
  gsl_matrix_free(vel_variance_new_quasi);
  gsl_matrix_free(denom);
  gsl_matrix_free(denom_temp);

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
  }else if(*MODEL_NUM == 5){
    *gi = bivariate_matern_salvana_multiple_advection_spacetime(PARAM_VECTOR, L1, L2);
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
