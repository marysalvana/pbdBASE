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

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>


#include <gsl/gsl_errno.h>

#define PI (3.141592653589793)
#define earthRadiusKm 6371.0

// This function converts decimal degrees to radians

static double deg2rad(double deg){
    return (deg * PI / 180);
}

//  This function converts radians to decimal degrees

static double rad2deg(double rad){
    return (rad * 180 / PI);
}

/*
* Returns the distance between two points on the Earth.
* Direct translation from http://en.wikipedia.org/wiki/Haversine_formula
* @param lat1d Latitude of the first point in degrees
* @param lon1d Longitude of the first point in degrees
* @param lat2d Latitude of the second point in degrees
* @param lon2d Longitude of the second point in degrees
* @return The distance between the two points in kilometers
*/

static double distanceEarth(double lat1d, double lon1d, double lat2d, double lon2d) {
    double lat1r, lon1r, lat2r, lon2r, u, v;
    lat1r = deg2rad(lat1d);
    lon1r = deg2rad(lon1d);
    lat2r = deg2rad(lat2d);
    lon2r = deg2rad(lon2d);
    u = sin((lat2r - lat1r)/2);
    v = sin((lon2r - lon1r)/2);
    //return 2.0 * earthRadiusKm * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
    return 2.0 * earthRadiusKm * sqrt(pow(u, 2) + cos(lat1r) * cos(lat2r) * pow(v, 2));
}

static double calculateDistance(double x1, double y1, double x2, double y2, int distance_metric) {
  if(distance_metric == 0){
    return  sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
  }else if(distance_metric == 1){
    return distanceEarth(x1, y1, x2, y2);
  }
}

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


int rmvnorm(const gsl_rng *r, const int n, const gsl_vector *mean, 
		const gsl_matrix *var, gsl_vector *result){
    /* multivariate normal distribution random number generator */
    /*
 *      *	n	dimension of the random vetor
 *           *	mean	vector of means of size n
 *                *	var	variance matrix of dimension n x n
 *                     *	result	output variable with a sigle random vector normal distribution generation
 *                          */
    int k;
    gsl_matrix *work = gsl_matrix_alloc(n,n);

    gsl_matrix_memcpy(work,var);

    gsl_linalg_cholesky_decomp(work);

    for(k=0; k<n; k++){
      gsl_vector_set(result, k, gsl_ran_ugaussian(r));
    }

    gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, var, result );
    gsl_vector_add(result,mean);
    gsl_matrix_free(work);
    return 0;
}

void InvertMatrix(const double m[16], double invOut[16])
{
    double inv[16], det;
    int i;

    inv[0] = m[5]  * m[10] * m[15] - 
             m[5]  * m[11] * m[14] - 
             m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] - 
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] + 
              m[4]  * m[11] * m[14] + 
              m[8]  * m[6]  * m[15] - 
              m[8]  * m[7]  * m[14] - 
              m[12] * m[6]  * m[11] + 
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] - 
             m[4]  * m[11] * m[13] - 
             m[8]  * m[5] * m[15] + 
             m[8]  * m[7] * m[13] + 
             m[12] * m[5] * m[11] - 
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] + 
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] - 
               m[8]  * m[6] * m[13] - 
               m[12] * m[5] * m[10] + 
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] + 
              m[1]  * m[11] * m[14] + 
              m[9]  * m[2] * m[15] - 
              m[9]  * m[3] * m[14] - 
              m[13] * m[2] * m[11] + 
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] - 
             m[0]  * m[11] * m[14] - 
             m[8]  * m[2] * m[15] + 
             m[8]  * m[3] * m[14] + 
             m[12] * m[2] * m[11] - 
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] + 
              m[0]  * m[11] * m[13] + 
              m[8]  * m[1] * m[15] - 
              m[8]  * m[3] * m[13] - 
              m[12] * m[1] * m[11] + 
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] - 
              m[0]  * m[10] * m[13] - 
              m[8]  * m[1] * m[14] + 
              m[8]  * m[2] * m[13] + 
              m[12] * m[1] * m[10] - 
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] - 
             m[1]  * m[7] * m[14] - 
             m[5]  * m[2] * m[15] + 
             m[5]  * m[3] * m[14] + 
             m[13] * m[2] * m[7] - 
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] + 
              m[0]  * m[7] * m[14] + 
              m[4]  * m[2] * m[15] - 
              m[4]  * m[3] * m[14] - 
              m[12] * m[2] * m[7] + 
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] - 
              m[0]  * m[7] * m[13] - 
              m[4]  * m[1] * m[15] + 
              m[4]  * m[3] * m[13] + 
              m[12] * m[1] * m[7] - 
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] + 
               m[0]  * m[6] * m[13] + 
               m[4]  * m[1] * m[14] - 
               m[4]  * m[2] * m[13] - 
               m[12] * m[1] * m[6] + 
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + 
              m[1] * m[7] * m[10] + 
              m[5] * m[2] * m[11] - 
              m[5] * m[3] * m[10] - 
              m[9] * m[2] * m[7] + 
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - 
             m[0] * m[7] * m[10] - 
             m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + 
             m[8] * m[2] * m[7] - 
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + 
               m[0] * m[7] * m[9] + 
               m[4] * m[1] * m[11] - 
               m[4] * m[3] * m[9] - 
               m[8] * m[1] * m[7] + 
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - 
              m[0] * m[6] * m[9] - 
              m[4] * m[1] * m[10] + 
              m[4] * m[2] * m[9] + 
              m[8] * m[1] * m[6] - 
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    det = 1.0 / det;

    for (i = 0; i < 16; i++){
      invOut[i] = inv[i] * det;
    }

 //   return 0;

}

double MatrixDeterminant(const double m[16])
{
    double inv[16], det;
    int i;

    inv[0] = m[5]  * m[10] * m[15] - 
             m[5]  * m[11] * m[14] - 
             m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] - 
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] + 
              m[4]  * m[11] * m[14] + 
              m[8]  * m[6]  * m[15] - 
              m[8]  * m[7]  * m[14] - 
              m[12] * m[6]  * m[11] + 
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] - 
             m[4]  * m[11] * m[13] - 
             m[8]  * m[5] * m[15] + 
             m[8]  * m[7] * m[13] + 
             m[12] * m[5] * m[11] - 
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] + 
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] - 
               m[8]  * m[6] * m[13] - 
               m[12] * m[5] * m[10] + 
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] + 
              m[1]  * m[11] * m[14] + 
              m[9]  * m[2] * m[15] - 
              m[9]  * m[3] * m[14] - 
              m[13] * m[2] * m[11] + 
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] - 
             m[0]  * m[11] * m[14] - 
             m[8]  * m[2] * m[15] + 
             m[8]  * m[3] * m[14] + 
             m[12] * m[2] * m[11] - 
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] + 
              m[0]  * m[11] * m[13] + 
              m[8]  * m[1] * m[15] - 
              m[8]  * m[3] * m[13] - 
              m[12] * m[1] * m[11] + 
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] - 
              m[0]  * m[10] * m[13] - 
              m[8]  * m[1] * m[14] + 
              m[8]  * m[2] * m[13] + 
              m[12] * m[1] * m[10] - 
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] - 
             m[1]  * m[7] * m[14] - 
             m[5]  * m[2] * m[15] + 
             m[5]  * m[3] * m[14] + 
             m[13] * m[2] * m[7] - 
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] + 
              m[0]  * m[7] * m[14] + 
              m[4]  * m[2] * m[15] - 
              m[4]  * m[3] * m[14] - 
              m[12] * m[2] * m[7] + 
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] - 
              m[0]  * m[7] * m[13] - 
              m[4]  * m[1] * m[15] + 
              m[4]  * m[3] * m[13] + 
              m[12] * m[1] * m[7] - 
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] + 
               m[0]  * m[6] * m[13] + 
               m[4]  * m[1] * m[14] - 
               m[4]  * m[2] * m[13] - 
               m[12] * m[1] * m[6] + 
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + 
              m[1] * m[7] * m[10] + 
              m[5] * m[2] * m[11] - 
              m[5] * m[3] * m[10] - 
              m[9] * m[2] * m[7] + 
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - 
             m[0] * m[7] * m[10] - 
             m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + 
             m[8] * m[2] * m[7] - 
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + 
               m[0] * m[7] * m[9] + 
               m[4] * m[1] * m[11] - 
               m[4] * m[3] * m[9] - 
               m[8] * m[1] * m[7] + 
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - 
              m[0] * m[6] * m[9] - 
              m[4] * m[1] * m[10] + 
              m[4] * m[2] * m[9] + 
              m[8] * m[1] * m[6] - 
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    return det;

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
  
  gsl_matrix_free(inv);
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

/*
 * Auxiliary functions for the bivariate_differential_operator_salvana_spatial
 * 
*/

double h(double scale_horizontal_space, double scale_vertical_space, 
  double lat1d, double lon1d, double pres1, 
  double lat2d, double lon2d, double pres2){

  double h_val = pow(scale_horizontal_space, 2) * pow(calculateDistance(lat1d, lon1d, lat2d, lon2d, 1), 2) +
    pow(scale_vertical_space, 2) * pow(pres1 - pres2, 2);  

  return h_val;
}

double h1(double scale_horizontal_space, double lat1d, double lon1d, double lat2d, double lon2d){

  double lat1r, lon1r, lat2r, lon2r, L, l, con;

  lat1r = deg2rad(lat1d);
  lon1r = deg2rad(lon1d);
  lat2r = deg2rad(lat2d);
  lon2r = deg2rad(lon2d);
  L = lat1r - lat2r;
  l = lon1r - lon2r;

  con = 4 * pow(scale_horizontal_space, 2) * pow(earthRadiusKm, 2);

  return con * (sin(L / 2) * cos(L / 2) - sin(lat1r) * cos(lat2r) * pow(sin(l / 2), 2));
}

double h3(double scale_horizontal_space, double lat1d, double lon1d, double lat2d, double lon2d){

  double lat1r, lon1r, lat2r, lon2r, L, l, con;

  lat1r = deg2rad(lat1d);
  lon1r = deg2rad(lon1d);
  lat2r = deg2rad(lat2d);
  lon2r = deg2rad(lon2d);
  l = lon1r - lon2r;

  con = 4 * pow(scale_horizontal_space, 2) * pow(earthRadiusKm, 2);

  //return con * cos(lat1r) * cos(lat2r) * sin(l / 2) * cos(l / 2);
  return con * sin(l / 2) * cos(l / 2);
}

double h33(double scale_horizontal_space, double lat1d, double lon1d, double lat2d, double lon2d){

  double lat1r, lon1r, lat2r, lon2r, L, l, con;

  lat1r = deg2rad(lat1d);
  lon1r = deg2rad(lon1d);
  lat2r = deg2rad(lat2d);
  lon2r = deg2rad(lon2d);
  l = lon1r - lon2r;

  con = 2 * pow(scale_horizontal_space, 2) * pow(earthRadiusKm, 2);

  //return con * cos(lat1r) * cos(lat2r) * (pow(cos(l / 2), 2) - pow(sin(l / 2), 2));
  return con * (pow(cos(l / 2), 2) - pow(sin(l / 2),2));
}

double h12(double scale_horizontal_space, double lat1d, double lon1d, double lat2d, double lon2d){

  double lat1r, lon1r, lat2r, lon2r, L, l, con;

  lat1r = deg2rad(lat1d);
  lon1r = deg2rad(lon1d);
  lat2r = deg2rad(lat2d);
  lon2r = deg2rad(lon2d);
  L = lat1r - lat2r;
  l = lon1r - lon2r;

  con = 4 * pow(scale_horizontal_space, 2) * pow(earthRadiusKm, 2);

  return con * (-pow(cos(L / 2), 2) / 2 + pow(sin(L / 2), 2) / 2 + sin(lat1r) * sin(lat2r) * pow(sin(l / 2), 2));
}

double h13(double scale_horizontal_space, double lat1d, double lon1d, double lat2d, double lon2d){

  double lat1r, lon1r, lat2r, lon2r, L, l, con;

  lat1r = deg2rad(lat1d);
  lon1r = deg2rad(lon1d);
  lat2r = deg2rad(lat2d);
  lon2r = deg2rad(lon2d);
  l = lon1r - lon2r;

  con = 4 * pow(scale_horizontal_space, 2) * pow(earthRadiusKm, 2);

  //return -con * sin(lat1r) * cos(lat2r) * sin(l / 2) * cos(l / 2);
  return -con * sin(lat1r) * sin(l / 2) * cos(l / 2);
}

double h4(double scale_vertical_space, double pres1, double pres2){
  return 2 * pow(scale_vertical_space, 2) * (pres1 - pres2);
}

double h44(double scale_vertical_space){
  return 2 * pow(scale_vertical_space, 2);
}

double C1(double scale_horizontal, double scale_vertical, double a1, double b1, double c1, double d1, double a2, double b2, double c2, double d2, double *l1, double *l2){

  double H = h(scale_horizontal, scale_vertical, l1[1], l1[0], l1[2], l2[1], l2[0], l2[2]);
  double H1 = h1(scale_horizontal, l1[1], l1[0], l2[1], l2[0]);
  double H2 = h1(scale_horizontal, l2[1], l1[0], l1[1], l2[0]);
  double H3 = h3(scale_horizontal, l1[1], l1[0], l2[1], l2[0]);
  double H4 = h4(scale_vertical, l1[2], l2[2]);


  double return_val = 0.25 * (a1 * a2 * H1 * H2 - b1 * b2 * pow(H3, 2) - c1 * c2 * pow(H4, 2) - a1 * b2 * H1 * H3
    + a2 * b1 * H2 * H3 - a1 * c2 * H1 * H4 + a2 * c1 * H2 * H4
    - b1 * c2 * H3 * H4 - b2 * c1 * H3 * H4) + H * d1 * d2;
int variable1 = l1[4];
int variable2 = l2[4];
 // printf("l1[0]: %f,l1[1]: %f,l1[2]: %f, variable1: %d,l2[0]: %f,l2[1]: %f,l2[2]: %f, variable2: %d,H: %f,H1: %f,H2: %f, H3: %f,H4: %f, return_val: %lf \n", l1[0], l1[1], l1[2], variable1, l2[0], l2[1], l2[2], variable2, H, H1, H2, H3, H4, return_val);

  return return_val;
}

double C2(double scale_horizontal, double scale_vertical, double smoothness, double a1, double b1, double c1, double d1, double a2, double b2, double c2, double d2, double *l1, double *l2){

  double H12 = h12(scale_horizontal, l1[1], l1[0], l2[1], l2[0]);
  double H13 = h13(scale_horizontal, l1[1], l1[0], l2[1], l2[0]);
  double H23 = h13(scale_horizontal, l2[1], l1[0], l1[1], l2[0]);
  double H33 = h33(scale_horizontal, l1[1], l1[0], l2[1], l2[0]);
  double H44 = h44(scale_vertical);

  //printf("l1[0]: %f,l1[1]: %f,l1[2]: %f,l2[0]: %f,l2[1]: %f,l2[2]: %f,H12: %f,H13: %f,H23: %f, H33: %f,H44: %f \n", l1[0], l1[1], l1[2], l2[0], l2[1], l2[2], H12, H13, H23, H33, H44);
  
  double return_val = -0.5 * (a1 * a2 * H12 - b1 * b2 * H33 - c1 * c2 * H44 - a1 * b2 * H13
    + a2 * b1 * H23) + 2 * smoothness * d1 * d2;
//add the smoothness term later
  return return_val;
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
  return cov_val;
}

double bivariate_matern_parsimonious_spatial(double *PARAM, double *l1, double *l2, int earth)
{
  double cov_val = 0.0;
  double expr = 0.0;
  double con = 0.0;
  double sigma_square = 0.0, range = PARAM[2], smoothness = 0.0;

  int variable1, variable2;

  variable1 = l1[4];
  variable2 = l2[4];

  if(variable1 == variable2){
    if(variable1 == 1){
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

  if(earth == 1){
    expr = sqrt(h(range, PARAM[6], l1[1], l1[0], l1[2], l2[1], l2[0], l2[2]));
  }else{
    expr = sqrt(pow(l1[0] - l2[0], 2) + pow(l1[1] - l2[1], 2)) / range;
  }
   
  if(expr == 0){
    cov_val = sigma_square;
  }else{

    gsl_set_error_handler_off();

    cov_val = con * pow(expr, smoothness) * gsl_sf_bessel_Knu(smoothness, expr);
  }
  return cov_val;
}

double univariate_matern_schlather_spacetime(double *PARAM, double *l1, double *l2, int vel_variance_supplied)
{
  double cov_val = 0.0;
  double expr = 0.0;
  double con = 0.0, con_new = 0.0;
  double sigma_square = PARAM[0], range = PARAM[1], smoothness = PARAM[2];
  double vel_mean_x = PARAM[3], vel_mean_y = PARAM[4];
  double vel_variance_chol_11 = 0.0, vel_variance_chol_12 = 0.0, vel_variance_chol_22 = 0.0;
  
  double l1x_new = 0.0, l1y_new = 0.0, l2x_new = 0.0, l2y_new = 0.0, xlag = 0.0, ylag = 0.0, tlag =0.0;
  double vel_variance11 = 0.0, vel_variance22 = 0.0, vel_variance12 = 0.0;
  double vel_variance11_new = 0.0, vel_variance22_new = 0.0, vel_variance12_new = 0.0;
  double Inv_11 = 0.0, Inv_22 = 0.0, Inv_12 = 0.0, det = 0.0;

  con = pow(2,(smoothness - 1)) * tgamma(smoothness);
  con = 1.0/con;

  l1x_new = l1[0] - vel_mean_x * l1[3];
  l1y_new = l1[1] - vel_mean_y * l1[3];
  l2x_new = l2[0] - vel_mean_x * l2[3];
  l2y_new = l2[1] - vel_mean_y * l2[3];

  xlag = l1x_new - l2x_new;
  ylag = l1y_new - l2y_new;
  tlag = l1[3] - l2[3];

  if(vel_variance_supplied == 0){

    vel_variance_chol_11 = PARAM[5]; 
    vel_variance_chol_12 = PARAM[6];
    vel_variance_chol_22 = PARAM[7];

    vel_variance11 = pow(vel_variance_chol_11, 2);
    vel_variance22 = pow(vel_variance_chol_12, 2) + pow(vel_variance_chol_22, 2);
    vel_variance12 = vel_variance_chol_11 * vel_variance_chol_12;

  }else if(vel_variance_supplied == 1){

    vel_variance11 = PARAM[5]; 
    vel_variance12 = PARAM[6];
    vel_variance22 = PARAM[7];

  }

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

double univariate_matern_numerical_lagrangian_spacetime(double *PARAM, double *l1, double *l2, int vel_variance_supplied)
{
  double cov_val = 0.0;
  double expr = 0.0;
  double con = 0.0, con_new = 0.0;
  double sigma_square = PARAM[0], range = PARAM[1], smoothness = PARAM[2];
  double vel_mean_x = PARAM[3], vel_mean_y = PARAM[4], vel_x = 0.0, vel_y = 0.0;
  double vel_variance_chol_11 = 0.0, vel_variance_chol_12 = 0.0, vel_variance_chol_22 = 0.0;
  
  double l1x_new = 0.0, l1y_new = 0.0, l2x_new = 0.0, l2y_new = 0.0, xlag = 0.0, ylag = 0.0, tlag = 0.0;
  double vel_variance11 = 0.0, vel_variance22 = 0.0, vel_variance12 = 0.0;

  unsigned long int seed = 0;

  const gsl_rng_type *T;
  gsl_rng *r; 

  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc(T);

  con = pow(2,(smoothness - 1)) * tgamma(smoothness);
  con = 1.0/con;

  gsl_vector *result = gsl_vector_alloc(2);
  gsl_vector *vel_mean = gsl_vector_alloc(2);
  gsl_matrix *vel_variance = gsl_matrix_alloc(2, 2);

  gsl_matrix_set_zero(vel_variance);

  gsl_vector_set(vel_mean, 0, vel_mean_x); 
  gsl_vector_set(vel_mean, 1, vel_mean_y); 

  if(vel_variance_supplied == 0){

    vel_variance_chol_11 = PARAM[5]; 
    vel_variance_chol_12 = PARAM[6];
    vel_variance_chol_22 = PARAM[7];

    vel_variance11 = pow(vel_variance_chol_11, 2);
    vel_variance22 = pow(vel_variance_chol_12, 2) + pow(vel_variance_chol_22, 2);
    vel_variance12 = vel_variance_chol_11 * vel_variance_chol_12;

  }else if(vel_variance_supplied == 1){

    vel_variance11 = PARAM[5]; 
    vel_variance12 = PARAM[6];
    vel_variance22 = PARAM[7];

  }

  gsl_matrix_set(vel_variance, 0, 0, vel_variance11); 
  gsl_matrix_set(vel_variance, 0, 1, vel_variance12); 
  gsl_matrix_set(vel_variance, 1, 0, vel_variance12); 
  gsl_matrix_set(vel_variance, 1, 1, vel_variance22); 

  for(seed=0; seed<100; seed++){

    gsl_rng_default_seed = seed + 1;
    rmvnorm(r, 2, vel_mean, vel_variance, result);

    vel_x = gsl_vector_get(result, 0);
    vel_y = gsl_vector_get(result, 1);

    l1x_new = l1[0] - vel_x * l1[2];
    l1y_new = l1[1] - vel_y * l1[2];
    l2x_new = l2[0] - vel_x * l2[2];
    l2y_new = l2[1] - vel_y * l2[2];

    xlag = l1x_new - l2x_new;
    ylag = l1y_new - l2y_new;

    expr = sqrt(pow(xlag, 2) + pow(ylag, 2)) / range;

    con_new = sigma_square * con;

    if(expr == 0){
      cov_val += sigma_square;
    }else{
      cov_val += con_new * pow(expr, smoothness) * gsl_sf_bessel_Knu(smoothness, expr);
    }
  }

  gsl_rng_free(r);
  gsl_vector_free(vel_mean);
  gsl_matrix_free(vel_variance);
  gsl_vector_free(result);

  return cov_val / 100;
}

double bivariate_matern_salvana_single_advection_spacetime(double *PARAM, double *l1, double *l2)
{
  double cov_val = 0.0;
  double expr = 0.0;
  double con = 0.0, con_new = 0.0;
  double sigma_square = 0.0, range = PARAM[2], smoothness = 0.0;
  int variable1, variable2;

  variable1 = l1[4];
  variable2 = l2[4];

  if(variable1 == variable2){
    if(variable1 == 1){
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

  l1x_new = l1[0] - vel_mean_x * l1[3];
  l1y_new = l1[1] - vel_mean_y * l1[3];
  l2x_new = l2[0] - vel_mean_x * l2[3];
  l2y_new = l2[1] - vel_mean_y * l2[3];

  xlag = l1x_new - l2x_new;
  ylag = l1y_new - l2y_new;
  tlag = l1[3] - l2[3];

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
  double hlag[2];
  double vel_mean_marginal[2], vel_mean[4], vel_mean_new[2];
  double T_matrix[8], T_matrix_squared[16];
  double param_vector[8];

  double vel_variance_marginal[3];
  double vel_variance_temp[16], vel_variance[16], vel_variance_inv[16], vel_variance_new[16], vel_variance_new_inv[16], denom[16];
  double vel_variance_new2[8], vel_variance_new3[4], vel_variance_new4[4];

  double sum = 0.0, xlag = 0.0, ylag = 0.0;
  double Inv_11 = 0.0, Inv_22 = 0.0, Inv_12 = 0.0;

  int i, j, k, t1, t2, variable1, variable2;

  hlag[0] = l1[0] - l2[0];
  hlag[1] = l1[1] - l2[1];

  t1 = l1[3];
  t2 = l2[3];
    
  variable1 = l1[4];
  variable2 = l2[4];

  vel_variance_temp[0] = pow(PARAM[10], 2);
  vel_variance_temp[5] = pow(PARAM[11], 2) + pow(PARAM[14], 2);
  vel_variance_temp[10] = pow(PARAM[12], 2) + pow(PARAM[15], 2) + pow(PARAM[17], 2);
  vel_variance_temp[15] = pow(PARAM[13], 2) + pow(PARAM[16], 2) + pow(PARAM[18], 2) + pow(PARAM[19], 2);

  vel_variance_temp[1] = vel_variance_temp[4] = PARAM[10] * PARAM[11];
  vel_variance_temp[2] = vel_variance_temp[8] = PARAM[10] * PARAM[12];
  vel_variance_temp[3] = vel_variance_temp[12] = PARAM[10] * PARAM[13];

  vel_variance_temp[6] = vel_variance_temp[9] = PARAM[11] * PARAM[12] + PARAM[14] * PARAM[15];
  vel_variance_temp[7] = vel_variance_temp[13] = PARAM[11] * PARAM[13] + PARAM[14] * PARAM[16];

  vel_variance_temp[11] = vel_variance_temp[14] = PARAM[12] * PARAM[13] + PARAM[15] * PARAM[16] + PARAM[17] * PARAM[18];

  if(variable1 == variable2){

    if(variable1 == 1){

      sigma_square = PARAM[0];
      smoothness = PARAM[3];
      vel_mean_marginal[0] = PARAM[6];
      vel_mean_marginal[1] = PARAM[7];
      vel_variance_marginal[0] = vel_variance_temp[0]; 
      vel_variance_marginal[1] = vel_variance_temp[1]; 
      vel_variance_marginal[2] = vel_variance_temp[5]; 
  
    }else{

      sigma_square = PARAM[1];
      smoothness = PARAM[4];
      vel_mean_marginal[0] = PARAM[8];
      vel_mean_marginal[1] = PARAM[9];
      vel_variance_marginal[0] = vel_variance_temp[10]; 
      vel_variance_marginal[1] = vel_variance_temp[11]; 
      vel_variance_marginal[2] = vel_variance_temp[15]; 
  
    }

    param_vector[0] = sigma_square;
    param_vector[1] = range;
    param_vector[2] = smoothness;
    param_vector[3] = vel_mean_marginal[0];
    param_vector[4] = vel_mean_marginal[1];
    param_vector[5] = vel_variance_marginal[0];
    param_vector[6] = vel_variance_marginal[1];
    param_vector[7] = vel_variance_marginal[2];

    cov_val = univariate_matern_schlather_spacetime(param_vector, l1, l2, 1);

  }else{

    sigma_square = PARAM[5] * sqrt(PARAM[0] * PARAM[1]);
    smoothness = 0.5 * (PARAM[3] + PARAM[4]);

    if(variable1 == 1 & variable2 == 2){

      vel_mean[0] = PARAM[6];
      vel_mean[1] = PARAM[7];
      vel_mean[2] = PARAM[8];
      vel_mean[3] = PARAM[9];

      for(i = 0; i < 4; i++){
        for(j = 0; j < 4; j++){
	  vel_variance[i * 4 + j] = vel_variance_temp[i * 4 + j];
        }
      }

    }else if(variable1 == 2 & variable2 == 1){

      vel_mean[0] = PARAM[8];
      vel_mean[1] = PARAM[9];
      vel_mean[2] = PARAM[6];
      vel_mean[3] = PARAM[7];

      vel_variance[0] = vel_variance_temp[10];
      vel_variance[5] = vel_variance_temp[15];
      vel_variance[10] = vel_variance_temp[0];
      vel_variance[15] = vel_variance_temp[5];
      vel_variance[1] = vel_variance[4] = vel_variance_temp[11];
      vel_variance[2] = vel_variance[8] = vel_variance_temp[8];
      vel_variance[3] = vel_variance[12] = vel_variance_temp[9];
      vel_variance[6] = vel_variance[9] = vel_variance_temp[12];
      vel_variance[7] = vel_variance[13] = vel_variance_temp[13];
      vel_variance[11] = vel_variance[14] = vel_variance_temp[1];

    }
  
    T_matrix[0] = T_matrix[5] = t1;
    T_matrix[2] = T_matrix[7] = -t2;

    T_matrix_squared[0] = T_matrix_squared[5] = pow(t1, 2);
    T_matrix_squared[10] = T_matrix_squared[15] = pow(t2, 2);
    T_matrix_squared[2] = T_matrix_squared[7] = T_matrix_squared[8] = T_matrix_squared[13] = -t1 * t2;
    T_matrix_squared[1] = T_matrix_squared[3] = T_matrix_squared[4] = T_matrix_squared[6] = 0.0;
    T_matrix_squared[9] = T_matrix_squared[11] = T_matrix_squared[12] = T_matrix_squared[14] = 0.0;
   
    InvertMatrix(vel_variance, vel_variance_inv);

    for(i = 0; i < 4; i++){
      for(j = 0; j < 4; j++){
        vel_variance_new[i * 4 + j] = T_matrix_squared[i * 4 + j] + vel_variance_inv[i * 4 + j];
      }
    }

    InvertMatrix(vel_variance_new, vel_variance_new_inv);

    for(k = 0; k < 2; k++){
      for(i = 0; i < 4; i++){
        for(j = 0; j < 4; j++){
          sum = sum + T_matrix[k * 4 + j] * vel_variance_new_inv[j * 4 + i];
        }
        vel_variance_new2[k * 4 + i] = sum;
        sum = 0.0;
      }
    }

    for(k = 0; k < 2; k++){
      for(i = 0; i < 2; i++){
        for(j = 0; j < 4; j++){
          sum = sum + vel_variance_new2[k * 4 + j] * T_matrix[i * 4 + j];
        }
        vel_variance_new3[k * 2 + i] = sum;
        sum = 0.0;
      }
    }

    vel_variance_new4[0] = 1 - vel_variance_new3[0];
    vel_variance_new4[1] = -vel_variance_new3[1];
    vel_variance_new4[2] = -vel_variance_new3[2];
    vel_variance_new4[3] = 1 - vel_variance_new3[3];

    for(k = 0; k < 2; k++){
      for(i = 0; i < 1; i++){
        for(j = 0; j < 4; j++){
          sum = sum + T_matrix[k * 4 + j] * vel_mean[i * 4 + j];
        }
        vel_mean_new[k + i] = sum;
        sum = 0.0;
      }
    }

    xlag = hlag[0] - vel_mean_new[0];
    ylag = hlag[1] - vel_mean_new[1];

    Inv_11 = vel_variance_new4[0];
    Inv_12 = vel_variance_new4[1];
    Inv_22 = vel_variance_new4[3];

    expr = sqrt(pow(xlag, 2) * Inv_11 + 2 * xlag * ylag * Inv_12 + pow(ylag, 2) * Inv_22 ) / range;

    for(k = 0; k < 4; k++){
      for(i = 0; i < 4; i++){
        for(j = 0; j < 4; j++){
          sum = sum + vel_variance[k * 4 + j] * T_matrix_squared[j * 4 + i];
        }
        if(k == i){
          denom[k * 4 + i] = 1 + sum;
        }else{
          denom[k * 4 + i] = sum;
        }
        sum = 0.0;
      }
    }

    det = MatrixDeterminant(denom);

    con = pow(2,(smoothness - 1)) * tgamma(smoothness);
    con = 1.0/con;

    con_new = sigma_square * con / sqrt(det);

    if(expr == 0){
      cov_val = sigma_square / sqrt(det);
    }else{
      cov_val = con_new * pow(expr, smoothness) * gsl_sf_bessel_Knu(smoothness, expr);
    }

  }

  //if(l1[0] == 0 & l1[1] == 0 & t1 == 0){
//    matrix_print(denom_temp);    
    //printf("expr: %f; det: %f \n", expr, det);
  //}

  return cov_val;

}

double bivariate_lmc_spatial(double *PARAM, double *l1, double *l2)
{
  double cov_val = 0.0, cov_val1 = 0.0, cov_val2 = 0.0;
  double sigma_square1 = 1.0, range1 = 0.0, smoothness1 = 0.0, paramvec1[3], sigma_square2 = 1.0, range2 = 0.0, smoothness2 = 0.0, paramvec2[3];
  double a11, a12, a21, a22;
  int variable1, variable2;

  variable1 = l1[3];
  variable2 = l2[3];

  range1 = PARAM[0];
  smoothness1 = PARAM[2];

  range2 = PARAM[1];
  smoothness2 = PARAM[3];
  
  paramvec1[0] = sigma_square1;
  paramvec1[1] = range1;
  paramvec1[2] = smoothness1;
  
  cov_val1 = univariate_matern_spatial(paramvec1, l1, l2);

  paramvec2[0] = sigma_square2;
  paramvec2[1] = range2;
  paramvec2[2] = smoothness2;

  cov_val2 = univariate_matern_spatial(paramvec2, l1, l2);

  a11 = PARAM[4];
  a12 = PARAM[5];
  a21 = PARAM[6];
  a22 = PARAM[7];

  if(variable1 == variable2){
    if(variable1 == 1){
      cov_val = pow(a11, 2) * cov_val1 + pow(a12, 2) * cov_val2;
    }else{
      cov_val = pow(a21, 2) * cov_val1 + pow(a22, 2) * cov_val2;
    }
  }else{
      cov_val = a11 * a21 * cov_val1 + a12 * a22 * cov_val2;
  }

  return cov_val;
}

double bivariate_lmc_salvana_spacetime(double *PARAM, double *l1, double *l2)
{
  double cov_val = 0.0, cov_val1 = 0.0, cov_val2 = 0.0;
  double sigma_square1 = 1.0, range1 = 0.0, smoothness1 = 0.0, sigma_square2 = 1.0, range2 = 0.0, smoothness2 = 0.0;
  double param_vector1[8], param_vector2[8], vel_mean_marginal1[2], vel_mean_marginal2[2], vel_variance_chol1[3], vel_variance_chol2[3];
  double a11 = 0.0, a12 = 0.0, a21 = 0.0, a22 = 0.0;
  int variable1, variable2;

  variable1 = l1[3];
  variable2 = l2[3];

  range1 = PARAM[0];
  smoothness1 = PARAM[2];

  range2 = PARAM[1];
  smoothness2 = PARAM[3];
  
  vel_mean_marginal1[0] = PARAM[4];
  vel_mean_marginal1[1] = PARAM[5];
  vel_mean_marginal2[0] = PARAM[6];
  vel_mean_marginal2[1] = PARAM[7];

  vel_variance_chol1[0] = PARAM[8];
  vel_variance_chol1[1] = PARAM[9];
  vel_variance_chol1[2] = PARAM[10];
  vel_variance_chol2[0] = PARAM[11];
  vel_variance_chol2[1] = PARAM[12];
  vel_variance_chol2[2] = PARAM[13];
  
  param_vector1[0] = sigma_square1;
  param_vector1[1] = range1;
  param_vector1[2] = smoothness1;
  param_vector1[3] = vel_mean_marginal1[0];
  param_vector1[4] = vel_mean_marginal1[1];
  param_vector1[5] = vel_variance_chol1[0];
  param_vector1[6] = vel_variance_chol1[1];
  param_vector1[7] = vel_variance_chol1[2];

  cov_val1 = univariate_matern_schlather_spacetime(param_vector1, l1, l2, 0);

  param_vector2[0] = sigma_square2;
  param_vector2[1] = range2;
  param_vector2[2] = smoothness2;
  param_vector2[3] = vel_mean_marginal2[0];
  param_vector2[4] = vel_mean_marginal2[1];
  param_vector2[5] = vel_variance_chol2[0];
  param_vector2[6] = vel_variance_chol2[1];
  param_vector2[7] = vel_variance_chol2[2];

  cov_val2 = univariate_matern_schlather_spacetime(param_vector2, l1, l2, 0);

  a11 = PARAM[14];
  a12 = PARAM[15];
  a21 = PARAM[16];
  a22 = PARAM[17];

  if(variable1 == variable2){
    if(variable1 == 1){
      cov_val = pow(a11, 2) * cov_val1 + pow(a12, 2) * cov_val2;
    }else{
      cov_val = pow(a21, 2) * cov_val1 + pow(a22, 2) * cov_val2;
    }
  }else{
      cov_val = a11 * a21 * cov_val1 + a12 * a22 * cov_val2;
  }

  return cov_val;
}

double univariate_matern_gneiting_spacetime(double *PARAM, double *l1, double *l2)
{
  double cov_val = 0.0;
  double expr = 0.0;
  double con = 0.0, con_new = 0.0;
  double sigma_square = PARAM[0], range_space = PARAM[1], smoothness = PARAM[2];
  double range_time = PARAM[3], alpha = PARAM[4], beta = PARAM[5], delta = PARAM[6];
  
  double xlag = 0.0, ylag = 0.0, tlag = 0.0, denom = 0.0;
  
  con = pow(2,(smoothness - 1)) * tgamma(smoothness);
  con = 1.0/con;

  xlag = l1[0] - l2[0];
  ylag = l1[1] - l2[1];
  tlag = l1[2] - l2[2];

  denom = pow(abs(tlag), 2 * alpha) / range_time + 1;

  expr = sqrt(pow(xlag, 2) + pow(ylag, 2)) / range_space;

  expr = expr / pow(denom, beta / 2);

  con_new = sigma_square * con / pow(denom, delta + beta);

  if(expr == 0){
    cov_val = sigma_square / pow(denom, delta + beta);
  }else{
    cov_val = con_new * pow(expr, smoothness) * gsl_sf_bessel_Knu(smoothness, expr);
  }
  return cov_val;
}

double bivariate_matern_bourotte_spacetime(double *PARAM, double *l1, double *l2)
{
  double cov_val = 0.0;
  double expr = 0.0;
  double con = 0.0, con_new = 0.0;
  double sigma_square = 0.0, range_space = PARAM[2], smoothness = 0.0;
  double range_time = PARAM[6], alpha = PARAM[7], beta = PARAM[8], delta = PARAM[9];
  double paramvec[7];
  double xlag = 0.0, ylag = 0.0, tlag = 0.0, denom = 0.0;

  int variable1, variable2;

  variable1 = l1[3];
  variable2 = l2[3];

  if(variable1 == variable2){
    if(variable1 == 1){
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

  paramvec[0] = sigma_square;
  paramvec[1] = range_space;
  paramvec[2] = smoothness;
  paramvec[3] = range_time;
  paramvec[4] = alpha;
  paramvec[5] = beta;
  paramvec[6] = delta;

  cov_val = univariate_matern_gneiting_spacetime(paramvec, l1, l2);

  return cov_val;
}

double univariate_deformation_matern_salvana_frozen_spacetime(double *PARAM, double *l1, double *l2)
{
  double cov_val = 0.0;
  double expr = 0.0;
  double con = 0.0, con_new = 0.0;
  double sigma_square = PARAM[0], range = PARAM[1], smoothness = PARAM[2];
  double vel_x = PARAM[3], vel_y = PARAM[4];
  
  double deform_source[2];
  double deform_coef_x[3], deform_coef_y[3];

  double dist_from_source_l1 = 0.0, dist_from_source_l2 = 0.0;  
  double l1x_new, l1y_new, l2x_new, l2y_new, l1x_deform, l1y_deform, l2x_deform, l2y_deform, xlag, ylag;
  
  deform_source[0] = PARAM[5];
  deform_source[1] = PARAM[6];

  deform_coef_x[0] = PARAM[7];
  deform_coef_x[1] = PARAM[8];
  deform_coef_x[2] = PARAM[9];

  deform_coef_y[0] = PARAM[10];
  deform_coef_y[1] = PARAM[11];
  deform_coef_y[2] = PARAM[12];

  con = pow(2,(smoothness - 1)) * tgamma(smoothness);
  con = 1.0/con;

  l1x_new = l1[0] - vel_x * l1[2];
  l1y_new = l1[1] - vel_y * l1[2];
  l2x_new = l2[0] - vel_x * l2[2];
  l2y_new = l2[1] - vel_y * l2[2];
  
  dist_from_source_l1 = sqrt(pow(l1x_new - deform_source[0], 2) + pow(l1y_new - deform_source[1], 2));
  dist_from_source_l2 = sqrt(pow(l2x_new - deform_source[0], 2) + pow(l2y_new - deform_source[1], 2));

  l1x_deform = deform_source[0] + (l1x_new - deform_source[0]) * (deform_coef_x[0] + deform_coef_x[1] * exp(-deform_coef_x[2] * pow(dist_from_source_l1, 2)));
  l1y_deform = deform_source[1] + (l1y_new - deform_source[1]) * (deform_coef_y[0] + deform_coef_y[1] * exp(-deform_coef_y[2] * pow(dist_from_source_l1, 2)));
  l2x_deform = deform_source[0] + (l2x_new - deform_source[0]) * (deform_coef_x[0] + deform_coef_x[1] * exp(-deform_coef_x[2] * pow(dist_from_source_l2, 2)));
  l2y_deform = deform_source[1] + (l2y_new - deform_source[1]) * (deform_coef_y[0] + deform_coef_y[1] * exp(-deform_coef_y[2] * pow(dist_from_source_l2, 2)));

  xlag = l1x_deform - l2x_deform;
  ylag = l1y_deform - l2y_deform;

  expr = sqrt(pow(xlag, 2) + pow(ylag, 2)) / range;

  con_new = sigma_square * con;

  if(expr == 0){
    cov_val = sigma_square;
  }else{
    cov_val = con_new * pow(expr, smoothness) * gsl_sf_bessel_Knu(smoothness, expr);
  }
  return cov_val;
}

double basis_function(double *PARAM, double *l1, double *l2)
{
  double cov_val = 0.0, expr = 0.0;
  double xlag, ylag;

  xlag = l1[0] - l2[0];
  ylag = l2[1] - l2[1];

  expr = sqrt(pow(xlag, 2) + pow(ylag, 2));

  if(expr == 0){
    cov_val = 0;
  }else{
    cov_val = pow(expr, 2) * log(expr);
  }
  return cov_val;
}



double bivariate_differential_operator_salvana_spatial(double *PARAM, double *l1, double *l2)
{
  double cov_val = 0.0;
  double expr = 0.0;
  double con = 0.0, f = 0.0, f_prime = 0.0, C1_val = 0.0, C2_val = 0.0;
  double sigma_square = 0.0, scale_horizontal = PARAM[2], scale_vertical = PARAM[3], smoothness = 0.0;
  double PARAM_SUB[11], a1 = 0.0, b1 = 0.0, c1 = 0.0, d1 = 0.0, a2 = 0.0, b2 = 0.0, c2 = 0.0, d2 = 0.0;

  int variable1, variable2;

  variable1 = l1[4];
  variable2 = l2[4];

  PARAM_SUB[0] = scale_horizontal;
  PARAM_SUB[1] = scale_vertical;

  if(variable1 == variable2){
    if(variable1 == 1){
      sigma_square = PARAM[0];
      smoothness = PARAM[4];
      PARAM_SUB[2] = PARAM_SUB[6] = PARAM[7];
      PARAM_SUB[3] = PARAM_SUB[7] = PARAM[8];
      PARAM_SUB[4] = PARAM_SUB[8] = PARAM[9];
      PARAM_SUB[5] = PARAM_SUB[9] = PARAM[10];
    }else{
      sigma_square = PARAM[1];
      smoothness = PARAM[5];
      PARAM_SUB[2] = PARAM_SUB[6] = PARAM[11];
      PARAM_SUB[3] = PARAM_SUB[7] = PARAM[12];
      PARAM_SUB[4] = PARAM_SUB[8] = PARAM[13];
      PARAM_SUB[5] = PARAM_SUB[9] = PARAM[14];
    }

  }else if(variable1 == 1 & variable2 == 2){
    sigma_square = PARAM[6] * sqrt(PARAM[0] * PARAM[1]);
    smoothness = 0.5 * (PARAM[4] + PARAM[5]);
    PARAM_SUB[2] = PARAM[7];
    PARAM_SUB[3] = PARAM[8];
    PARAM_SUB[4] = PARAM[9];
    PARAM_SUB[5] = PARAM[10];
    PARAM_SUB[6] = PARAM[11];
    PARAM_SUB[7] = PARAM[12];
    PARAM_SUB[8] = PARAM[13];
    PARAM_SUB[9] = PARAM[14];
  }else if(variable1 == 2 & variable2 == 1){
    sigma_square = PARAM[6] * sqrt(PARAM[0] * PARAM[1]);
    smoothness = 0.5 * (PARAM[4] + PARAM[5]);
    PARAM_SUB[2] = PARAM[11];
    PARAM_SUB[3] = PARAM[12];
    PARAM_SUB[4] = PARAM[13];
    PARAM_SUB[5] = PARAM[14];
    PARAM_SUB[6] = PARAM[7];
    PARAM_SUB[7] = PARAM[8];
    PARAM_SUB[8] = PARAM[9];
    PARAM_SUB[9] = PARAM[10];
  }
  PARAM_SUB[10] = smoothness;

  con = pow(2, (smoothness - 1)) * tgamma(smoothness);
  con = 1.0/con;
  con = sigma_square * con;
 
  expr = sqrt(h(PARAM_SUB[0], PARAM_SUB[1], l1[1], l1[0], l1[2], l2[1], l2[0], l2[2]));

  C1_val = C1(scale_horizontal, scale_vertical, a1, b1, c1, d1, a2, b2, c2, d2, l1, l2);
  C2_val = C2(scale_horizontal, scale_vertical, smoothness, a1, b1, c1, d1, a2, b2, c2, d2, l1, l2);

  if(expr == 0){
    cov_val = con * (C1_val + C2_val) + sigma_square * d1 * d2;
  }else{
    gsl_set_error_handler_off();
    
    f = pow(expr, smoothness - 1) * gsl_sf_bessel_Knu(smoothness - 1, expr);
    f_prime = pow(expr, smoothness - 2) * gsl_sf_bessel_Knu(smoothness - 2, expr);
  
    cov_val = con * (C1_val * f_prime + C2_val * f + d1 * d2 * (pow(expr, 2) * f_prime + 2 * (smoothness - 1) * f));
    //cov_val = con * (C1(PARAM_SUB, l1, l2) * pow(expr, smoothness - 1) * gsl_sf_bessel_Knu(smoothness - 1, expr)
      //+ C2(PARAM_SUB, l1, l2) * pow(expr, smoothness) * gsl_sf_bessel_Knu(smoothness, expr));
  }
  return cov_val;
}

double bivariate_differential_operator_with_bsplines_salvana_spatial(double *PARAM, double *l1, double *l2)
{
  double cov_val = 0.0, cov_val_temp = 0.0;
  double expr = 0.0;
  double con = 0.0, f = 0.0, f_prime = 0.0, C1_val = 0.0, C2_val = 0.0;
  double sigma_square = 0.0, scale_horizontal = PARAM[2], scale_vertical = PARAM[3], smoothness = 0.0;
  double PARAM_SUB[11], a1 = 0.0, b1 = 0.0, c1 = 0.0, d1 = 0.0, a2 = 0.0, b2 = 0.0, c2 = 0.0, d2 = 0.0;

  int variable1 = l1[4], variable2 = l2[4];

  if(variable1 == variable2){
    if(variable1 == 1){
      sigma_square = PARAM[0];
      smoothness = PARAM[4];
      a1 = a2 = PARAM[7];
      b1 = b2 = PARAM[8];
      d1 = d2 = PARAM[9];
      c1 = l1[5];
      c2 = l2[5];
    }else{
      sigma_square = PARAM[1];
      smoothness = PARAM[5];
      a1 = a2 = PARAM[10];
      b1 = b2 = PARAM[11];
      d1 = d2 = PARAM[12];
      c1 = l1[6];
      c2 = l2[6];
    }

  }else if(variable1 == 1 & variable2 == 2){
    sigma_square = PARAM[6] * sqrt(PARAM[0] * PARAM[1]);
    smoothness = 0.5 * (PARAM[4] + PARAM[5]);
    a1 = PARAM[7];
    b1 = PARAM[8];
    d1 = PARAM[9];
    c1 = l1[5];
    a2 = PARAM[10];
    b2 = PARAM[11];
    d2 = PARAM[12];
    c2 = l2[6];
  }else if(variable1 == 2 & variable2 == 1){
    sigma_square = PARAM[6] * sqrt(PARAM[0] * PARAM[1]);
    smoothness = 0.5 * (PARAM[4] + PARAM[5]);
    a1 = PARAM[10];
    b1 = PARAM[11];
    d1 = PARAM[12];
    c1 = l1[6];
    a2 = PARAM[7];
    b2 = PARAM[8];
    d2 = PARAM[9];
    c2 = l2[5];
  }

  con = pow(2, (smoothness - 1)) * tgamma(smoothness);
  con = 1.0/con;
  con = sigma_square * con;
 
  expr = sqrt(h(scale_horizontal, scale_vertical, l1[1], l1[0], l1[2], l2[1], l2[0], l2[2]));

  C1_val = C1(scale_horizontal, scale_vertical, a1, b1, c1, d1, a2, b2, c2, d2, l1, l2);
  C2_val = C2(scale_horizontal, scale_vertical, smoothness, a1, b1, c1, d1, a2, b2, c2, d2, l1, l2);
  //C2_val = C2(PARAM_SUB, l1, l2);

  double nugget = 1e-3;
  if(expr == 0){
    cov_val = con * (C1_val + C2_val) + sigma_square * d1 * d2;
    if(variable1 == variable2){
      cov_val = cov_val + nugget;
    }
  }else{

    gsl_set_error_handler_off();
    
    f = pow(expr, smoothness - 1) * gsl_sf_bessel_Knu(smoothness - 1, expr);
    f_prime = pow(expr, smoothness - 2) * gsl_sf_bessel_Knu(smoothness - 2, expr);
  
    cov_val = con * (C1_val * f_prime + C2_val * f + d1 * d2 * (pow(expr, 2) * f_prime + 2 * (smoothness - 1) * f));
    //cov_val = con * ((C1(PARAM_SUB, l1, l2) + pow(expr, 2) * d1 * d2) * pow(expr, smoothness - 2) * gsl_sf_bessel_Knu(smoothness - 2, expr)
      //+ (C2(PARAM_SUB, l1, l2) + 2 * (smoothness - 1) * d1 * d2) * pow(expr, smoothness - 1) * gsl_sf_bessel_Knu(smoothness - 1, expr));
  }
  
  if(variable1 == 2 & variable2 == 2){
    //printf("l1[0]: %f, l1[1]: %f, l1[2]: %f, l2[0]: %f, l2[1]: %f, l2[2]: %f, C1_val: %lf, C2_val: %lf, \n", l1[0], l1[1], l1[2], l2[0], l2[1], l2[2], C1_val, C2_val);
  //printf("c1: %lf, c2: %lf, l1[2]: %f,l2[2]: %f,f: %lf, f_prime: %lf, C1_val: %lf, C2_val: %lf, expr: %lf, cov_val: %lf, \n", c1, c2, l1[2], l2[2], f, f_prime, C1_val, C2_val, expr, cov_val);
  }

  return cov_val;
}

void covfunc_(int *MODEL_NUM, double *PARAM_VECTOR, double *L1, double *L2, double *gi)
{

  if(*MODEL_NUM == 0){
    *gi = basis_function(PARAM_VECTOR, L1, L2);
  }else if(*MODEL_NUM == 1){
    *gi = univariate_matern_spatial(PARAM_VECTOR, L1, L2);
  }else if(*MODEL_NUM == 2){
    *gi = univariate_matern_schlather_spacetime(PARAM_VECTOR, L1, L2, 0);
  }else if(*MODEL_NUM == 3){
    *gi = bivariate_matern_parsimonious_spatial(PARAM_VECTOR, L1, L2, 1);
  }else if(*MODEL_NUM == 4){
    *gi = bivariate_matern_salvana_single_advection_spacetime(PARAM_VECTOR, L1, L2);
  }else if(*MODEL_NUM == 5){
    *gi = bivariate_matern_salvana_multiple_advection_spacetime(PARAM_VECTOR, L1, L2);
  }else if(*MODEL_NUM == 6){
    *gi = bivariate_lmc_spatial(PARAM_VECTOR, L1, L2);
  }else if(*MODEL_NUM == 7){
    *gi = univariate_deformation_matern_salvana_frozen_spacetime(PARAM_VECTOR, L1, L2);
  }else if(*MODEL_NUM == 8){
    *gi = bivariate_lmc_salvana_spacetime(PARAM_VECTOR, L1, L2);
  }else if(*MODEL_NUM == 9){
    *gi = univariate_matern_gneiting_spacetime(PARAM_VECTOR, L1, L2);
  }else if(*MODEL_NUM == 10){
    *gi = bivariate_matern_bourotte_spacetime(PARAM_VECTOR, L1, L2);
  }else if(*MODEL_NUM == 11){
    *gi = univariate_matern_numerical_lagrangian_spacetime(PARAM_VECTOR, L1, L2, 0);
  }else if(*MODEL_NUM == 12){
    *gi = bivariate_differential_operator_salvana_spatial(PARAM_VECTOR, L1, L2);
  }else if(*MODEL_NUM == 13){
    *gi = bivariate_differential_operator_with_bsplines_salvana_spatial(PARAM_VECTOR, L1, L2);
  }else if(*MODEL_NUM == 14){
    *gi = bivariate_matern_parsimonious_spatial(PARAM_VECTOR, L1, L2, 1);
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
