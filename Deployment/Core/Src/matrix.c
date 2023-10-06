#include "matrix.h"

// ------------------------ Matrix Function ------------------------

Matrix *Matrix_gen(int row, int column, MATRIX_TYPE *data) {/*
 * Generate a new Matrix(struct).
 * 导入_生成矩阵*/
    Matrix *_mat = NULL;
    _mat = (Matrix *) malloc(sizeof(Matrix));
    _mat->row = row;
    _mat->column = column;
    _mat->data = data;
    return _mat;
}

Matrix *M_add(Matrix *_mat_add1, Matrix *_mat_add2) {/*
 * Addition/ Subtraction between Matrix-s (create).
 * 矩阵加减法 */

    int size = (_mat_add1->row) * (_mat_add1->column), i;
    for (i = 0; i < size; i++) {
        _mat_add1->data[i] = _mat_add1->data[i] + _mat_add2->data[i];
    }
    return _mat_add1;
}

Matrix *M_mul(Matrix *_mat_left, Matrix *_mat_right) {/*
 * Matrix multiplication (create new one, abbr. create).
 * _mat_result = _mat_left*_mat_right
 * 矩阵乘法 */
    /*Determine_Matrix_Dimensions*/
    Matrix *_mat_result = NULL;
    if (_mat_left->column != _mat_right->row) {
        printf(M_mul_001);
    } else {
        _mat_result = (Matrix *) malloc(sizeof(Matrix));
        int row = _mat_left->row;
        int mid = _mat_left->column;
        int column = _mat_right->column;
        int i, j, k;
        MATRIX_TYPE *_data = (MATRIX_TYPE *) malloc((row * column) * sizeof(MATRIX_TYPE));
        _mat_result->row = row;
        _mat_result->column = column;
        _mat_result->data = _data;
        // MATRIX_TYPE *_data =  _mat_result->data;
        MATRIX_TYPE temp = 0;
        /*Ergodic*/
        for (i = 0; i < row; i++) {
            for (j = 0; j < column; j++) {
                /*Caculate Element*/
                temp = 0;
                for (k = 0; k < mid; k++) {
                    temp += (_mat_left->data[i * mid + k]) * (_mat_right->data[k * column + j]);
                }
                _data[i * column + j] = temp;
            }
        }
    }
    return _mat_result;
}


Matrix *M_relu(Matrix *_mat) {/*
 * Addition/ Subtraction between Matrix-s (create).
 * 矩阵relu */
    int size = (_mat->row) * (_mat->column), i;
    for (i = 0; i < size; i++) {
        _mat->data[i] = _mat->data[i] < 0.0 ? 0.0 : _mat->data[i];
    }
    return _mat;
}


// int M_print(Matrix *_mat) {/*
//  * Matrix Print, Display.
//  * 打印矩阵 */
//     printf(">>Matrix_%x:\n", _mat);
//     int i, j;
//     for (i = 0; i < _mat->row; i++) {
//         for (j = 0; j < _mat->column; j++) {
//             printf(PRECISION, _mat->data[i * (_mat->column) + j]);
//         }
//         printf("\n");
//     }
//     return 0;
// }