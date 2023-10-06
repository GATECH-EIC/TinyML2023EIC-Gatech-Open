#ifndef __MATRIX_H
#define __MATRIX_H


#include <stdio.h>
#include <math.h>
#include <string.h>

#define MATRIX_TYPE float
/* The Level of Details of Output - 显示详细等级*/
#define M_mul_001        "@ERROR: Matrix_Dimensions Wrong!\n\tDetails:(M_mul_001)_mat_left->column != _mat_right->row\n"
#define M_add_sub_003    "@ERROR: Matrix_Dimensions Wrong!\n\tDetails:(M_add_sub_003)_mat_add1 != _mat_add2\n"
#define PRECISION "%.3f\t"

typedef struct _Matrix {
    /*Store Matrix
	存储矩阵*/
    int row;
    int column;
    MATRIX_TYPE *data;
} Matrix;


Matrix *Matrix_gen(int row, int column, MATRIX_TYPE *data);

Matrix *M_mul(Matrix *_mat_left, Matrix *_mat_right);

Matrix *M_add(Matrix *_mat_add1, Matrix *_mat_add2);

Matrix *M_relu(Matrix *_mat);

int M_print(Matrix *_mat);


#endif