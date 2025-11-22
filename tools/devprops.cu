#include <cstdio>
#include <cuda_runtime.h>
int main(){
  int dev=0; cudaDeviceProp p;
  if(cudaGetDeviceProperties(&p,dev)!=cudaSuccess){fprintf(stderr,"fail\n");return 1;}
  int clock_kHz = 0;
  if (cudaDeviceGetAttribute(&clock_kHz, cudaDevAttrClockRate, dev) != cudaSuccess) {
    fprintf(stderr,"fail_getattr\n");
    return 1;
  }
  printf("name=%s\nmultiProcessorCount=%d\nmajor=%d\nminor=%d\nclockRate_kHz=%d\n",
         p.name,p.multiProcessorCount,p.major,p.minor,clock_kHz);
  return 0;
}