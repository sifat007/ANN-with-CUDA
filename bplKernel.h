#ifndef __BPLKERNEL__
#define __BPLKERNEL__

static void HandleError( cudaError_t err,
                        const char *file,
                        int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void *myCudaMalloc(size_t len)
{
    void *p;
    HANDLE_ERROR(cudaMalloc(&p, len));
    return p;
}


void initializeW(double* X1, long A, long B);

void initializeI(double* X1, long A, long B);

void initializeO(double* X1, long A, long B);


void mm(double* X2, double* Y, double* Z1, long A, long B, long C);

void mtm(double* X2, double* Y1, double* Z1, long A, long B, long C);

void mmt(double* X1, double* Y2, double* Z1, long A, long B, long C);

void func(double* X1, double* Yf, long A, long B1, long val);

void gradient_func(double* X1, double* Yf, long A, long B);

void error(double* X1, double* Y, double* Z,  long A, long B);

void reduction(double* Y, double* X1, long A, long B);

void prob(double* Y,double* Z, double* X1, long A, long B);

void delta(double* Z, double* Y, long A, long B, double C);







#endif
