/*---------------------------------------------------------------------------------------------------------------*/
/// bpl.c
/// For CSU CS475 Fall 2016
/// Instructor: Sanjay Rajopadhye
/// GTA: Swetha Varadarajan
/// Based on code Created by Paul Tero at Existor Ltd as part of a neural networks tutorial
/// Modified by Swetha Varadarajan
/// Created: 2016-11-16
/*---------------------------------------------------------------------------------------------------------------*/

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <math.h>

#include "timer.h"
#include "util.h"
#include "bplKernel.h"

#define X(i, j) X[((i) * (cmdLineArgs.N + 1)) + (j)]
#define H(i, j) H[((i) * (cmdLineArgs.M + 1)) + (j)]




int main(int argc, char * * argv) {

    /*---------------------------------------------------------------------------------------------------------------*/
    /*-----------------------------------------Command line parsing--------------------------------------------------*/
    /*---------------------------------------------------------------------------------------------------------------*/

    Params cmdLineArgs;
    parseCmdLineArgs( & cmdLineArgs, argc, argv);

    /*---------------------------------------------------------------------------------------------------------------*/
    /*-------------------------------------------Variable Declaration------------------------------------------------*/
    /*---------------------------------------------------------------------------------------------------------------*/

    /*Array description and its size in the comments next to its declation*/

    double * inputs; //Given inputs = total number of samples(S)*number of inputs per sample(N)
    double * outputs; //Expected outputs = total number of samples(S)*number of outputs per sample(P)

    double * X; //Input for a given iteration = bunch size(I)*number of inputs per sample(N+1(bias))
    double * Y; //Output for a given iteration = bunch size(I)*number of outputs per sample(P)

    double * Wxh; //Weights in between input and hidden layer = (N+1)*M
    double * Why; //Weights in between input and hidden layer = (M+1)*P
    double * dWxh; //Error Weights in between input and hidden layer = (N+1)*M
    double * dWhy; //Error Weights in between input and hidden layer = (M+1)*P

    double * Zh; //Weighted sum for hidden layer=I*M
    double * H; // Activation values = I*(M+1)
    double * Zy; //Weighted sum for output layer=I*P
    double * E; //Calculated Errors = I*P
    double * P1; //Oredicted output = I*P
    double * P; // (exp(Zy)) = I*P
    double * sum; //(summation of the P[i]s) = I
    
    
    
    double * d_X; //Input for a given iteration = bunch size(I)*number of inputs per sample(N+1(bias))
    double * d_Y; //Output for a given iteration = bunch size(I)*number of outputs per sample(P)

    double * d_Wxh; //Weights in between input and hidden layer = (N+1)*M
    double * d_Why; //Weights in between input and hidden layer = (M+1)*P
    double * d_dWxh; //Error Weights in between input and hidden layer = (N+1)*M
    double * d_dWhy; //Error Weights in between input and hidden layer = (M+1)*P

    double * d_Zh; //Weighted sum for hidden layer=I*M
    double * d_H; // Activation values = I*(M+1)
    double * d_Zy; //Weighted sum for output layer=I*P
    double * d_E; //Calculated Errors = I*P
    double * d_P1; //Oredicted output = I*P
    double * d_P; // (exp(Zy)) = I*P
    double * d_sum; //(summation of the P[i]s) = I
    
    

    double learningrate = 0.0001; /*learning rate */
    long b = cmdLineArgs.sample_per_iter;

    long k2 = cmdLineArgs.sample_total / b; /*number of full bunches */
    long k3 = cmdLineArgs.sample_total - (k2 * b); /* size of the partial bunch */

    /*---------------------------------------------------------------------------------------------------------------*/
    /*-------------------------------------------Memory allocations--------------------------------------------------*/
    /*---------------------------------------------------------------------------------------------------------------*/


    
    inputs = (double * ) malloc(cmdLineArgs.sample_total * sizeof(double) * cmdLineArgs.N);
    outputs = (double * ) malloc(cmdLineArgs.sample_total * sizeof(double) * cmdLineArgs.P);


    sum = (double * ) malloc((b) * sizeof(double));    

    Wxh = (double * ) malloc((cmdLineArgs.N + 1) * sizeof(double) * cmdLineArgs.M);
    Why = (double * ) malloc((cmdLineArgs.M + 1) * sizeof(double) * cmdLineArgs.P);
    dWxh = (double * ) malloc((cmdLineArgs.N + 1) * sizeof(double) * cmdLineArgs.M);
    dWhy = (double * ) malloc((cmdLineArgs.M + 1) * sizeof(double) * cmdLineArgs.P);

    X = (double * ) malloc(b * sizeof(double) * (cmdLineArgs.N + 1));
    Y = (double * ) malloc((b) * sizeof(double) * cmdLineArgs.P);    
    E = (double * ) malloc(b * sizeof(double) * (cmdLineArgs.P));
    P = (double * ) malloc(b * sizeof(double) * (cmdLineArgs.P));
    P1 = (double * ) malloc(b * sizeof(double) * (cmdLineArgs.P));
    H = (double * ) malloc(b * sizeof(double) * (cmdLineArgs.M + 1));
    Zh = (double * ) malloc(b * sizeof(double) * (cmdLineArgs.M));
    Zy = (double * ) malloc(b * sizeof(double) * (cmdLineArgs.P));


    d_sum = (double * ) myCudaMalloc((b) * sizeof(double));

        
    d_Wxh = (double * ) myCudaMalloc((cmdLineArgs.N + 1) * sizeof(double) * cmdLineArgs.M);
    d_Why = (double * ) myCudaMalloc((cmdLineArgs.M + 1) * sizeof(double) * cmdLineArgs.P);
    d_dWxh = (double * ) myCudaMalloc((cmdLineArgs.N + 1) * sizeof(double) * cmdLineArgs.M);
    d_dWhy = (double * ) myCudaMalloc((cmdLineArgs.M + 1) * sizeof(double) * cmdLineArgs.P);
        
    d_X = (double * ) myCudaMalloc(b * sizeof(double) * (cmdLineArgs.N + 1));
    d_Y = (double * ) myCudaMalloc((b) * sizeof(double)* cmdLineArgs.P);    
    d_E = (double * ) myCudaMalloc(b * sizeof(double) * (cmdLineArgs.P));
    d_P = (double * ) myCudaMalloc(b * sizeof(double) * (cmdLineArgs.P));
    d_P1 = (double * ) myCudaMalloc(b * sizeof(double) * (cmdLineArgs.P));
    d_H = (double * ) myCudaMalloc(b * sizeof(double) * (cmdLineArgs.M + 1));
    d_Zh = (double * ) myCudaMalloc(b * sizeof(double) * (cmdLineArgs.M));
    d_Zy = (double * ) myCudaMalloc(b * sizeof(double) * (cmdLineArgs.P));
        
        

    if (inputs == NULL || outputs == NULL || X == NULL || H == NULL || dWxh == NULL || dWhy == NULL || Zh == NULL || Zy == NULL || Wxh == NULL || Why == NULL || E == NULL || P == NULL || P1 == NULL || sum == NULL) {
        printf("Could not allocate memory\n");
        exit(0);
    }
    /*---------------------------------------------------------------------------------------------------------------*/
    /*----------------------------------------------Initializations--------------------------------------------------*/
    /*---------------------------------------------------------------------------------------------------------------*/

    initializeW(Wxh, (cmdLineArgs.N + 1), cmdLineArgs.M);
    initializeW(Why, (cmdLineArgs.M + 1), cmdLineArgs.P);
    initializeI(inputs, cmdLineArgs.sample_total, cmdLineArgs.N);
    initializeO(outputs, cmdLineArgs.sample_total, cmdLineArgs.P);
        
        //displayMatrix1("outputs", outputs, cmdLineArgs.P, cmdLineArgs.sample_total);
        
    HANDLE_ERROR(cudaMemcpy(d_Wxh, Wxh, (cmdLineArgs.N + 1) * sizeof(double) * cmdLineArgs.M, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_Why, Why, (cmdLineArgs.M + 1) * sizeof(double) * cmdLineArgs.P, cudaMemcpyHostToDevice));
        


    /*---------------------------------------------------------------------------------------------------------------*/
    /*------------------------------------------------Training-------------------------------------------------------*/
    /*---------------------------------------------------------------------------------------------------------------*/
    initialize_timer();
    start_timer();

    for (long t = 0; t < cmdLineArgs.iter; t++) //Time loop
    {
        for (long s = 0; s < k2; s++) //Bunch loop
        {
                        
            HANDLE_ERROR(cudaMemcpy(H, d_H, b * sizeof(double) * (cmdLineArgs.M+1), cudaMemcpyDeviceToHost)); //
                        
            for (long i = 0; i < b; i++) {
                X(i, 0) = H(i, 0) = 1; //bias setting
                //required input/output are copied from inputs/outputs to X and Y
                memcpy( & X(i, 1), & inputs[cmdLineArgs.N * ((s * b) + i)], cmdLineArgs.N * sizeof(double));
            }
            Y = & outputs[s * b * cmdLineArgs.P];
            cudaMemcpy(d_X, X, b * sizeof(double) * (cmdLineArgs.N + 1), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Y, Y, b * sizeof(double) * cmdLineArgs.P, cudaMemcpyHostToDevice);                        
            cudaMemcpy(d_H, H, b * sizeof(double) * (cmdLineArgs.M+1), cudaMemcpyHostToDevice); //set bias on device
                        
                        
                        
            /*Forward Phase*/

            mm(d_Zh, d_X, d_Wxh, b, cmdLineArgs.N + 1, cmdLineArgs.M); //Zh=X*Wxh			
            func(d_H, d_Zh, b, cmdLineArgs.M, 1); //H=f1(Zh)                        
            mm(d_Zy, d_H, d_Why, b, cmdLineArgs.M + 1, cmdLineArgs.P); //Zy=H*Why                        
            func(d_P, d_Zy, b, cmdLineArgs.P, 0); //P=fn(Zy)                        
            reduction(d_P, d_sum, b, cmdLineArgs.P); //summation of probabilities for each training sample                                                
            prob(d_P, d_P1, d_sum, b, cmdLineArgs.P); //P1=fn(P,sum)                        
            error(d_E, d_P1, d_Y, b, cmdLineArgs.P); //E=P1-Y

            /*Backprpagation Phase*/
            
            

            mtm(d_dWhy, d_H, d_E, cmdLineArgs.M + 1, b, cmdLineArgs.P); //dWhy=H'*E ('->transpose)                        
            delta(d_Why, d_dWhy, cmdLineArgs.M + 1, cmdLineArgs.P, learningrate); //Why=fn(dwhy)                        
            mmt(d_H, d_Why, d_E, b, cmdLineArgs.M + 1, cmdLineArgs.P); //H=Why*E'                        
            gradient_func(d_Zh, d_H, b, cmdLineArgs.M); //Zh=f1"(H) ("->gradient of f1)                        
            mtm(d_dWxh, d_X, d_Zh, cmdLineArgs.N + 1, b, cmdLineArgs.M); //dWxh=X'Zh                        
            delta(d_Wxh, d_dWxh, cmdLineArgs.N + 1, cmdLineArgs.M, learningrate); //Wxh=fn(dWxh)
                        
                        
        }
        
        if (k3) {
            //printf("k3: %ld\n", k3);
            HANDLE_ERROR(cudaMemcpy(H, d_H, k3 * sizeof(double) * (cmdLineArgs.M+1), cudaMemcpyDeviceToHost)); //
                        
            for (long i = 0; i < k3; i++) {
                X(i, 0) = H(i, 0) = 1;
                memcpy( & X(i, 1), & inputs[cmdLineArgs.N * ((k2 * b) + i)], cmdLineArgs.N * sizeof(double));
            }
            Y = & outputs[k2 * b * cmdLineArgs.P];
            cudaMemcpy(d_X, X, k3 * sizeof(double) * (cmdLineArgs.N + 1), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Y, Y, k3 * sizeof(double) * cmdLineArgs.P, cudaMemcpyHostToDevice);                        
            cudaMemcpy(d_H, H, k3 * sizeof(double) * (cmdLineArgs.M+1), cudaMemcpyHostToDevice); //set bias on device
            


            //Forward Phase
            mm(d_Zh, d_X, d_Wxh, k3, cmdLineArgs.N + 1, cmdLineArgs.M);                     
            func(d_H, d_Zh, k3, cmdLineArgs.M, 1);            
            mm(d_Zy, d_H, d_Why, k3, cmdLineArgs.M + 1, cmdLineArgs.P);            
            func(d_P, d_Zy, k3, cmdLineArgs.P, 0);
            reduction(d_P, d_sum, k3, cmdLineArgs.P);                      
            prob(d_P, d_P1, d_sum, k3, cmdLineArgs.P);            
            error(d_E, d_P1, d_Y, k3, cmdLineArgs.P);                       

            //Backprpagation Phase
            mtm(d_dWhy, d_H, d_E, cmdLineArgs.M + 1, k3, cmdLineArgs.P);
            delta(d_Why, d_dWhy, cmdLineArgs.M + 1, cmdLineArgs.P, learningrate);
            mmt(d_H, d_Why, d_E, k3, cmdLineArgs.M + 1, cmdLineArgs.P);
            gradient_func(d_Zh, d_H, k3, cmdLineArgs.M);
            mtm(d_dWxh, d_X, d_Zh, cmdLineArgs.N + 1, k3, cmdLineArgs.M);
            delta(d_Wxh, d_dWxh, cmdLineArgs.N + 1, cmdLineArgs.M, learningrate);
                        
                        

        }
        
    }

    stop_timer();
    double time = elapsed_time();
    
    
    double nFlops = b* (cmdLineArgs.N + 1)* cmdLineArgs.M*2;
    nFlops += b* cmdLineArgs.M;
    nFlops += b *(cmdLineArgs.M + 1) * cmdLineArgs.P*2;
    nFlops += b * cmdLineArgs.P * 4;  //for 4 different kernels;
    nFlops += (cmdLineArgs.M + 1)* b * cmdLineArgs.P*2;
    nFlops += (cmdLineArgs.M + 1) * cmdLineArgs.P *2;
    nFlops += b* (cmdLineArgs.M + 1)* cmdLineArgs.P *2;
    nFlops += b * cmdLineArgs.M;
    nFlops += (cmdLineArgs.N + 1)* b * cmdLineArgs.M *2;
    nFlops += (cmdLineArgs.N + 1)* cmdLineArgs.M * 2;
    
    nFlops *= cmdLineArgs.iter * k2;
    
    nFlops += k3* (cmdLineArgs.N + 1)* cmdLineArgs.M*2;
    nFlops += k3* cmdLineArgs.M;
    nFlops += k3 *(cmdLineArgs.M + 1) * cmdLineArgs.P*2;
    nFlops += k3 * cmdLineArgs.P * 4;  //for 4 different kernels;
    nFlops += (cmdLineArgs.M + 1)* k3 * cmdLineArgs.P*2;
    nFlops += (cmdLineArgs.M + 1) * cmdLineArgs.P *2;
    nFlops += k3* (cmdLineArgs.M + 1)* cmdLineArgs.P *2;
    nFlops += k3 * cmdLineArgs.M;
    nFlops += (cmdLineArgs.N + 1)* k3 * cmdLineArgs.M *2;
    nFlops += (cmdLineArgs.N + 1)* cmdLineArgs.M * 2;
    
    
    
    double nFlopsPerSec = nFlops/time;
    double nGFlopsPerSec = nFlopsPerSec*1e-9;
    
    
    printf("Time: %lf\n", time);
    //printf("Time: %lf  GFlops: %lf\n", time, nGFlopsPerSec);
        
    cudaDeviceSynchronize();    
        

    
    //get results from the device to host
    HANDLE_ERROR(cudaMemcpy(Wxh, d_Wxh, (cmdLineArgs.N + 1) * sizeof(double) * cmdLineArgs.M, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(Why, d_Why, (cmdLineArgs.M + 1) * sizeof(double) * cmdLineArgs.P, cudaMemcpyDeviceToHost));
    
    /*---------------------------------------------------------------------------------------------------------------*/
    /*----------------------------------------------Print outputs----------------------------------------------------*/
    /*---------------------------------------------------------------------------------------------------------------*/
    if (cmdLineArgs.V) {
        /*Need the following 2 statements for Testing*/
        displayMatrix1("input/hidden weights", Wxh, cmdLineArgs.N + 1, cmdLineArgs.M);
        displayMatrix1("hidden/output weights", Why, cmdLineArgs.M + 1, cmdLineArgs.P);
        /* Useful for analyzing the accuracy of prediction */
        /*if(k3)
        {
                displayVector ("last input", &X[k3-1][1], cmdLineArgs.N);
                displayVector ("last output", Y[k3-1], cmdLineArgs.P);
                displayVector ("predicted output",P1[k3-1], cmdLineArgs.P);
        }
        else
        {
                displayVector ("last input", &X[b-1][1], cmdLineArgs.N);
                displayVector ("last output", Y[b-1], cmdLineArgs.P);
                displayVector ("predicted output",P1[b-1], cmdLineArgs.P);
        }*/
    }
    /*---------------------------------------------------------------------------------------------------------------*/
    /*----------------------------------------------Free Memory------------------------------------------------------*/
    /*---------------------------------------------------------------------------------------------------------------*/
    free(inputs);
    free(outputs);
    free(X);
    free(Zh);
    free(Zy);
    free(H);
    free(E);
    free(P);
    free(P1);
    free(sum);
    free(Wxh);
    free(Why);
    free(dWxh);
    free(dWhy);
        
        
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Zh);
    cudaFree(d_Zy);
    cudaFree(d_H);
    cudaFree(d_E);
    cudaFree(d_P);
    cudaFree(d_P1);
    cudaFree(d_sum);
    cudaFree(d_Wxh);
    cudaFree(d_Why);
    cudaFree(d_dWxh);
    cudaFree(d_dWhy);
        
        
    /*-------------------------------------------------------END-----------------------------------------------------*/
    return 0;
}
