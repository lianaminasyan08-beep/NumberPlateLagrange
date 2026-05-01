#include <cmath>
#include <vector>

extern "C" {

// =========================
// LAGRANGE INTERPOLATION
// =========================
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

// =========================
// SIGNATURE FUNCTION (USED BY PYTHON)
// =========================
void compute_signature(double* nums, int n, double* out) {

    std::vector<double> x(n);
    std::vector<double> y(n);

    for (int i = 0; i < n; i++) {
        x[i] = i;
        y[i] = nums[i];
    }

    int out_size = 60;

    // Generate interpolated points
    for (int i = 0; i < out_size; i++) {
        double t = (double)i * (n - 1) / (out_size - 1);
        out[i] = lagrange_eval(x.data(), y.data(), n, t);
    }

    // =========================
    // NORMALIZATION (same as Python)
    // =========================
    double mean = 0.0;
    for (int i = 0; i < out_size; i++) {
        mean += out[i];
    }
    mean /= out_size;

    double std = 0.0;
    for (int i = 0; i < out_size; i++) {
        out[i] -= mean;
        std += out[i] * out[i];
    }
    std = sqrt(std / out_size) + 1e-9;

    double norm = 0.0;
    for (int i = 0; i < out_size; i++) {
        out[i] /= std;
        norm += out[i] * out[i];
    }
    norm = sqrt(norm) + 1e-9;

    for (int i = 0; i < out_size; i++) {
        out[i] /= norm;
    }
}

}