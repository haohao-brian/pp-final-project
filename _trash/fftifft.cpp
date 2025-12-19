#include <cstdio>
#include <cstdlib>
#include <complex>
#include <cstring>
#include "fft.h"
using namespace std;

#define DEBUG 1

int main(int argc, char **argv) {
    int n;
    complex<double> *in = NULL;
    complex<double> *out = NULL;
    if (DEBUG) {
        n = 1 << 20;
        in = (complex<double>*) calloc(n, sizeof(complex<double>));
        out = (complex<double>*) calloc(n, sizeof(complex<double>));

        srand(time(NULL));
        for (len_t i = 0; i < n; i++) {
            in[i] = complex<double>(i, 0);
        }
    } else {
        n = strlen(argv[1]);
        in = (complex<double>*) calloc(n, sizeof(complex<double>));
        out = (complex<double>*) calloc(n, sizeof(complex<double>));
        for (int i = 0; i < n; i++)
            in[i] = complex<double>(argv[1][i] - '0', 0);
    }
    int num_threads = 1;
    fft_plan plan = fft_plan_dft_1d(n, in, out, false, num_threads, false, false);
    for(int i = 0; i < 1; i++) fft_execute(plan);
    fft_destroy_plan(plan);
    
    plan = fft_plan_dft_1d(n, out, in, true, num_threads, false, false);
    for(int i = 0; i < 1; i++) fft_execute(plan);
    fft_destroy_plan(plan);
    
    for (int i = 0; i < 10; i++) printf("%d", (int)(in[i].real() + 0.5));
    puts("");
    free(in);
    free(out);
    return 0;
}