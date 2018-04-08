#include<stdio.h>


#define Y1(i,j) Y1[((i)*(A))+(j)]
#define Yf(i,j) Yf[((i)*(B1))+(j)]
#define Y2(i,j) Y2[((i)*(C))+(j)]
#define Z1(i,j) Z1[((i)*(C))+(j)]
#define X1(i,j) X1[((i)*(B))+(j)]
#define X2(i,j) X2[((i)*(C))+(j)]
#define Y(i,j) Y[((i)*(B))+(j)]
#define Z(i,j) Z[((i)*(B))+(j)]
//#define I(i,j) I[((i)*(A))+(j)]
#define foo(a,b) b?tanh(a):exp(a)


#define FOOTPRINT_SIZE 64
#define BLOCK_SIZE 32
#define THREADS_PER_BLOCK 32 //for Pointwise calculations


void *myCudaMalloc1(size_t len)
{
    void *p;
    cudaMalloc(&p, len);
    return p;
}

void displayMatrix2 (const char *label, double *m, int rows, int cols)
{
printf ("\n%s:\n", label);
for(int i = 0; i < rows; ++i )
{
    for(int j = 0; j < cols; ++j )
            printf("%10.5lf\t",m[(i*cols)+j]);
                printf ("\n");
    }
    
}


__global__ void MatMulKernel(double* C, double* A, double* B, int A_width, int A_height, int B_width, int B_height, bool transA, bool transB);
//__global__ void MatMulKernel01(double* C, double* A, double* B, int A_width, int A_height, int B_width, int B_height, bool transA, bool transB);

__global__ void cuMinus(double *C, double *A, double *B, int n, double delta=1);

__global__ void cuGradientFunc(double *A, double *B, long n, long n_cols);

__global__ void cuFunc(double *A, double *B, long n, long n_cols, long val);

__global__ void cuDivideByVec(double *C, double *A, double *B, long n, long n_cols);

__global__ void cu_sum(const double* src, double* sum, double *global_mem, const int n);


//---------------------------Helper Host Functions------------------------------------------------------------------------------------------------
void initializeW(double* X1, long A, long B){
    /*Initializes the weights*/
    long i,j;
    for (i=0; i<A;i++)
        for (j=0; j<B;j++)
            X1(i,j) = ((double)rand() / (double)RAND_MAX) * 0.2 - 0.1;

}

void initializeI(double* X1, long A, long B){
    /*Initializes the inputs*/
    long i,j;
    for (i=0; i<A;i++)
        for (j=0; j<B;j++)
            X1(i,j) = j%2;

}

void initializeO(double* X1, long A, long B){
    /*Initializes the outputs*/
    
    long i,j;
    for (i=0; i<A;i++)
        for (j=0; j<B;j++)
            X1(i,j) = i%2;

}


void mm(double* X2, double* Y, double* Z1, long A, long B, long C){
    /*Performs Matrix-Matrix Mulitplication*/
    /*
    long i,j,k;
    for (i=0; i<A; i++) 
        for (j=0; j<B; j++)
            for(k=0; k<C; k++) 
            {
                if(j==0) X2(i,k)=0;
                X2(i,k) += Y(i,j) * Z1(j,k);
            }
    */

    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    bool transA = false, transB = false;        
    int A_width = B;
    int A_height = A;
    int B_width = C;
    int B_height = B;
    //printf("%dx%d %dx%d\n", A_height, A_width, B_height, B_width);
    int grid_size_x = transB? ((B_height-1)/BLOCK_SIZE + 1) : ((B_width-1)/BLOCK_SIZE + 1);
    int grid_size_y = transA? ((A_width-1)/BLOCK_SIZE + 1) : ((A_height-1)/BLOCK_SIZE + 1);
    dim3 dimGrid( grid_size_x, grid_size_y);
    MatMulKernel<<<dimGrid,dimBlock>>>(X2,Y,Z1,A_width,A_height,B_width,B_height, transA,transB);
    
}



void mmt(double* X1, double* Y2, double* Z1, long A, long B, long C){
    /*Performs Matrix-Transposed Matrix Mulitplication*/
    
    /*
    long i,j,k;
    for (i=0; i<A; i++) 
        for (j=0; j<B; j++)
        {
            X1(i,j)=0;
            for(k=0; k<C; k++)
                X1(i,j) += Z1(i,k) * Y2(j,k) ;   //Z1(i,k)
        }
    */
    
    
    //printf("%d %d %d\n",A,B,C);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    bool transA = false, transB = true;        
    int A_width = C;
    int A_height = A;
    int B_width = C;
    int B_height = B;
    //printf("%dx%d %dx%d\n", A_height, A_width, B_height, B_width);
    int grid_size_x = transB? ((B_height-1)/BLOCK_SIZE + 1) : ((B_width-1)/BLOCK_SIZE + 1);
    int grid_size_y = transA? ((A_width-1)/BLOCK_SIZE + 1) : ((A_height-1)/BLOCK_SIZE + 1);
    dim3 dimGrid( grid_size_x, grid_size_y);
    MatMulKernel<<<dimGrid,dimBlock>>>(X1,Z1,Y2,A_width,A_height,B_width,B_height, transA,transB);
    
    
}




void mtm(double* X2, double* Y1, double* Z1, long A, long B, long C){
    /*Performs Transposed Matrix- Matrix Mulitplication*/
    
    /*
    long i,j,k;
    for (i=0; i<A; i++) 
        for (j=0; j<B; j++)
            for(k=0; k<C; k++)
            { 
                if(j==0) X2(i,k)=0;
                X2(i,k) += Y1(j,i) * Z1(j,k);
            }
    */

    
    //printf("%d %d %d\n",A,B,C);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    bool transA = true, transB = false;        
    int A_width = A;
    int A_height = B;
    int B_width = C;
    int B_height = B;
    //printf("%dx%d %dx%d\n", A_height, A_width, B_height, B_width);
    int grid_size_x = transB? ((B_height-1)/BLOCK_SIZE + 1) : ((B_width-1)/BLOCK_SIZE + 1);
    int grid_size_y = transA? ((A_width-1)/BLOCK_SIZE + 1) : ((A_height-1)/BLOCK_SIZE + 1);
    dim3 dimGrid( grid_size_x, grid_size_y);
    MatMulKernel<<<dimGrid,dimBlock>>>(X2,Y1,Z1,A_width,A_height,B_width,B_height, transA,transB);
    
}




void func(double* X1, double* Yf, long A, long B1, long val){
    /*Performs a point-wise operation*/
    long B=B1+val;
/*	long i,j;
    for (i=0; i<A; i++) 
        for (j=0; j<B1; j++)
            X1(i,(j+val)) = foo(Yf(i,j),val); */
    long len = A*B1;
    const size_t block_size = THREADS_PER_BLOCK;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cuFunc<<<num_blocks, block_size>>>(X1, Yf, len, B1, val);
}


void gradient_func(double* X1, double* Yf, long A, long B){
    /*Performs a point-wise operation*/
    long B1=B+1;
    /*
    long i,j;
    for (i=0; i<A; i++)
        for (j=0; j<B; j++)  
            X1(i,j) = Yf(i, (j+1))*(1 - pow (tanh (X1(i,j)), 2)); 
    */
    
    long len = A*B;
    const size_t block_size = THREADS_PER_BLOCK;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cuGradientFunc<<<num_blocks, block_size>>>(X1, Yf, len, B);
    
}



void error(double* X1, double* Y, double* Z,  long A, long B){
    /*Calculates the Error*/
    /*
    long i,j;
    for (i=0; i<A; i++)
        for (j=0; j<B; j++)
            X1(i,j) = Y(i,j)-Z(i,j); 
    */
    
        long len = A*B;
    const size_t block_size = THREADS_PER_BLOCK;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cuMinus<<<num_blocks, block_size>>>(X1, Y, Z, len);
}


void reduction(double* Y, double* X1, long A, long B){
    /*Performs the summation of probabilities*/
    /*long i,j;
    for (i=0; i<A; i++)
    {
        X1[i]=0;
        for (j=0; j<B; j++)
            X1[i] += Y(i,j);
    }*/
    
    int len = B;
    const size_t block_size = THREADS_PER_BLOCK;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    
    double * HostX = (double * ) malloc(A * sizeof(double));
            
    double *data;
    double *d_partial_sums; 
    double *global_mem;
    data = (double * ) myCudaMalloc1(sizeof(double) * len);
    global_mem = (double * ) myCudaMalloc1( sizeof(double)* block_size);  
    d_partial_sums = (double * ) myCudaMalloc1( sizeof(double)* num_blocks);  
        
    for(int i = 0; i < A; ++i){
        int tmp_block_size = block_size;
        int tmp_num_blocks = num_blocks;
        int data_len = len;
        cudaMemcpy(data, Y + i * len, data_len * sizeof(double), cudaMemcpyDeviceToDevice);
        while(true){
            cudaMemset(global_mem, 0, sizeof(double) * tmp_block_size);
            cu_sum<<<tmp_num_blocks, tmp_block_size>>>(data, d_partial_sums, global_mem, data_len);
                        cudaDeviceSynchronize();
            data_len = tmp_num_blocks;
            if(tmp_num_blocks == 1){
                // copy the result back to the host
                double host_res = 0;
                cudaMemcpy(&host_res, d_partial_sums, sizeof(double), cudaMemcpyDeviceToHost);
                HostX[i] = host_res;
                break;
            }else if(tmp_num_blocks <= block_size){
                tmp_block_size = data_len;
                tmp_num_blocks = 1;
                cudaMemcpy(data, d_partial_sums, data_len * sizeof(double), cudaMemcpyDeviceToDevice);
            }else{
                tmp_block_size = THREADS_PER_BLOCK;
                tmp_num_blocks = (data_len / tmp_block_size) + ((data_len % tmp_block_size) ? 1 : 0);
                cudaMemcpy(data, d_partial_sums, data_len * sizeof(double), cudaMemcpyDeviceToDevice);
            }
        }
    } 
    
    cudaMemcpy(X1, HostX, A * sizeof(double), cudaMemcpyHostToDevice); //copy back to the device
        
    cudaFree(global_mem);
    cudaFree(data);
    cudaFree(d_partial_sums);
    free(HostX);
    //displayMatrix2("HostX", HostX, A, 1);
        
    
}


void prob(double* Y,double* Z, double* X1, long A, long B){
    /*Computes the normalized exponential*/
    /*long i,j;
    for (i=0; i<A; i++)
        for (j=0; j<B; j++)
            Z(i,j) = Y(i,j)/X1[i];*/
        long len = A*B;
    const size_t block_size = THREADS_PER_BLOCK;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cuDivideByVec<<<num_blocks, block_size>>>(Z, Y, X1, len, B);
    
}


void delta(double* Z, double* Y, long A, long B, double C){
    /*Updates the weight matrix*/
    /*
    long i,j;
    for (i=0; i<A; i++)
        for (j=0; j<B; j++) 
            Z(i,j) -= C*Y(i,j); 
    */
        
    long len = A*B;
    const size_t block_size = THREADS_PER_BLOCK;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cuMinus<<<num_blocks, block_size>>>(Z, Z, Y, len, C);
}


//----------------Device kernels---------------------------------

__global__ void MatMulKernel(double* C, double* A, double* B, int A_width, int A_height, int B_width, int B_height, bool transA, bool transB)
{
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int Row = block_row * BLOCK_SIZE + thread_row,
        Col = block_col * BLOCK_SIZE + thread_col;
        

    int C_width = transB?B_height:B_width;
    int C_height = transA?A_width:A_height;
    
    
    //if(transB && !block_col && !block_row && !thread_col && !thread_row)printf("C: %d %d\n",C_width, C_height);


    float Cvalue = 0;

    for (int m = 0;  m < (transA?A_height:A_width - 1) / BLOCK_SIZE + 1; ++m) {

        __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

        
        if(transA){
            if(BLOCK_SIZE * m + thread_col < A_height && Row < A_width) {
                shared_A[thread_row][thread_col] = A[(BLOCK_SIZE * m + thread_col) * A_width + Row];
            }else{
                shared_A[thread_row][thread_col] = 0;
            }
        }else{
            if(Row < A_height && BLOCK_SIZE * m + thread_col < A_width) {
                shared_A[thread_row][thread_col] = A[Row * A_width + BLOCK_SIZE * m + thread_col];
            }else{
                shared_A[thread_row][thread_col] = 0;
            }
        }
        
        if(transB){
            if( Col < B_height && BLOCK_SIZE * m + thread_row < B_width) {
                shared_B[thread_row][thread_col] = B[ Col * B_width + BLOCK_SIZE * m + thread_row];
            } else {
                shared_B[thread_row][thread_col] = 0;
            }
        }else{
            if(BLOCK_SIZE * m + thread_row < B_height && Col < B_width ) {
                shared_B[thread_row][thread_col] = B[ (BLOCK_SIZE * m + thread_row) * B_width + Col];
            } else {
                shared_B[thread_row][thread_col] = 0;
            }
        }
        
        
        // Synchronize to ensure all elements are read
        __syncthreads();

        #pragma unroll
        for(int e=0; e<BLOCK_SIZE; ++e)
            Cvalue += shared_A[thread_row][e] * shared_B[e][thread_col];
        __syncthreads();
    }

    if(Row < C_height && Col < C_width) {
        C[Row * C_width + Col] = Cvalue;
    }
}





__global__ void cuMinus(double *C, double *A, double *B, int n, double delta){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while(tid < n){
            if(delta != 1){
                C[tid] = A[tid] - B[tid] * delta;
            }else{
                C[tid] = A[tid] - B[tid];
            }
            tid += stride;
    }
}



__global__ void cuDivideByVec(double *C, double *A, double *B, long n, long n_cols){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while(tid < n){
        C[tid] = A[tid] / B[tid/n_cols];
        tid += stride;
    }
}



__global__ void cuGradientFunc(double *A, double *B, long n, long n_cols){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while(tid < n){
        A[tid] =  (1 - pow (tanh (A[tid]), 2)) * B[tid+1 + (tid)/n_cols];
        tid += stride;
    }
}

__global__ void cuFunc(double *A, double *B, long n, long n_cols, long val){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while(tid < n){
        A[tid+ val*(val+tid/n_cols)] =  foo(B[tid],val);
        tid += stride;
    }
}





__global__ void cu_sum(const double* src, double* sum, double *global_mem, const int n){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // load input into __shared__ memory
        //for(int i = 0 ; i < n; i++)printf("%lf ",src[i]);
    // printf("\n");
    double x = 0;
    if(tid < n){
        x = src[tid];
    }
    global_mem[threadIdx.x] = x;
    __syncthreads();
    // contiguous range pattern
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1){
        if(threadIdx.x < offset){
            // add a partial sum upstream to our own
            global_mem[threadIdx.x] += global_mem[threadIdx.x + offset];
        }
        // wait until all threads in the block have
        // updated their partial sums
        __syncthreads();
    }
    // thread 0 writes the final result
    if(threadIdx.x == 0){
        sum[blockIdx.x] = global_mem[0];
    }
    __syncthreads();
}
