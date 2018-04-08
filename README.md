## ANN-with-CUDA
Pure C and CUDA used to write a GPU parallelized version of Artificial Neural Network.

## Usage
Once you make the files in the respective folders, running the executable with -H option 
gives the following:
```
./bpl -H
usage: ./bpl
  -N Number of input layer neurons
  -M Number of hidden layer neurons
  -P Number of output layer neurons
  -S Number of training samples
  -I Number of training samples per bunch
  -L Number of iterations of time loop
  -V Verbose ON
  -H usage help, this dialogue
```
The N,M,P,S,I,L options require arguments that are greater than or equal to 1.

Note: Number of training samples per iteration (I) should be less than or equal to 
  Number of training samples (S)
  
