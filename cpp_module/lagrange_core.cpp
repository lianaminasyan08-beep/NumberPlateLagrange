#include <cmath>
#include <stdio.h>

extern "C" {

double lagrange_eval(double* x, double* y, int n, double t) {
    double result = 0.0;

    for (int i = 0; i < n; i++) {
        double term = y[i];

        for (int j = 0; j < n; j++) {
            if (i != j) {
                term *= (t - x[j]) / (x[i] - x[j]);
            }
        }

        result += term;
    }

    return result;
}

}