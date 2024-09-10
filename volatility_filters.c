#include <math.h>
#include <stdio.h>

//gcc -shared -o volatility_filters.so volatility_filters.c -fPIC -lm

void ngarch_filter(const float *log_xret, float *h, float *eps, int n_days,
                   float h_min, float lmbda, float omega, float alpha, float beta, float gamma) {
    for (int tn = 0; tn < n_days; tn++) {
        eps[tn] = (log_xret[tn] + 0.5 * h[tn]) / sqrt(h[tn]) - lmbda;
        //printf("tn=%d -- r,h,eps: %f, %f, %f\n",tn, log_xret[tn], h[tn], eps[tn]);
        h[tn + 1] = omega + alpha * h[tn] * (eps[tn] - gamma) * (eps[tn] - gamma) + beta * h[tn];
        if (h[tn + 1] < h_min) {
            h[tn + 1] = h_min;
        }
    }
}

void ngarch_sim(const float *z, float *ex_r, float *h, int n_days, int n_paths,
                    float lmbda, float omega, float alpha, float beta, float gamma) {
    
    for (int tn = 0; tn < n_days; tn++) {
        for (int path = 0; path < n_paths; path++) {
            float sig = sqrt(h[tn * n_paths + path]);

            ex_r[tn * n_paths + path] = lmbda * sig - 0.5 * h[tn * n_paths + path] + sig * z[tn * n_paths + path];

            h[(tn + 1) * n_paths + path] = omega + alpha * h[tn * n_paths + path] * (z[tn * n_paths + path] - gamma) * (z[tn * n_paths + path] - gamma) + beta * h[tn * n_paths + path];

            // if (tn < 2 && path < 2) {
            //   printf("tn=%d pn=%d -- r,h,eps: %f, %f, %f\n",tn,path,
            //          z[tn * n_paths + path], ex_r[tn * n_paths + path], h[(tn + 1) * n_paths + path]);
            //   fflush(stdout);
            // }
        }
    }
}
