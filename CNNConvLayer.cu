// This program executes a typical convolutional layer in regular CNNs.Neuron sparsity(zero ratio) is 50% and Weight sparsity is 70%.
#include <iostream>
#include "CNNConvLayer.h"
using namespace std;

int *filtCooNNZ_GPU;
int *filtCooData_GPU;
int *filtCooRow_GPU;
int *filtCooCol_GPU;

int *inNeuCooNNZ_GPU;
int *inNeuCooData_GPU;
int *inNeuCooRow_GPU;
int *inNeuCooCol_GPU;

int *tem_out_GPU;
int *out_GPU;
// This is the CPU version, please don't modify it
void convLayerCPU()
{
	// declarations for bunch of indexing parameters
	int fn, sli, fmy, fmx, y, x;
	int ifmy, ifmx, ofmy, ofmx;
	int filtIdx, inNeuIdx, outNeuIdx, outIdx;
	int filtVol  = FMDEPTH  * FILTSIZE * FILTSIZE;
	int fmArea   = FMSIZE   * FMSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int outArea  = FMSIZE/3 * FMSIZE/3;
	int sum;
	// Convolution
	for(fn = 0; fn < FILTNUM; fn++){
		for(fmy = 0; fmy < FMSIZE; fmy += STRIDE){
			for(fmx = 0; fmx < FMSIZE; fmx += STRIDE){
				sum = 0;
				for(sli = 0; sli < FMDEPTH; sli++){
					for(y = 0; y < FILTSIZE; y++){
						for(x = 0; x < FILTSIZE; x++){
							ifmy = fmy - FILTSIZE / 2 + y;
							ifmx = fmx - FILTSIZE / 2 + x;
							filtIdx = fn*filtVol + sli*filtArea + y*FILTSIZE + x;
							inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
							if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
								sum += filt[filtIdx] * inNeu[inNeuIdx];
						}
					}
				}
				// Activation - ReLU
				outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;
				if(sum <= 0)
					outNeu[outNeuIdx] = 0;
				else
					outNeu[outNeuIdx] = sum;
			}
		}
	}

	// Max Pooling with Window Size 3x3 and stride 3
	int max, tmpVal;
	for(sli = 0; sli < FILTNUM; sli++){
		for(fmy = 0; fmy < FMSIZE/3 ; fmy += 1){
			for(fmx = 0; fmx < FMSIZE/3 ; fmx += 1){
				outNeuIdx = sli*fmArea + fmy*3*FMSIZE + fmx*3;
				max = outNeu[outNeuIdx];
				for(y = 0; y < 3; y++){
					for(x = 0; x < 3; x++){
						ofmy = fmy*3 + y;
						ofmx = fmx*3 + x;
						outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
						tmpVal = outNeu[outNeuIdx];	
						if(tmpVal > max)
							max = tmpVal;
					}
				}
				outIdx = sli*outArea + fmy*FMSIZE/3 + fmx;
				outCPU[outIdx] = max;
			}
		}
	}
}
void init_Mem_GPU(){
	cudaMalloc(&filtCooNNZ_GPU , FILTNUM*FMDEPTH* sizeof(int));
	cudaMalloc(&filtCooData_GPU, filtCooSize    * sizeof(int)); 
	cudaMalloc(&filtCooRow_GPU , filtCooSize    * sizeof(int));
	cudaMalloc(&filtCooCol_GPU , filtCooSize    * sizeof(int));

	cudaMalloc(&inNeuCooNNZ_GPU , FMDEPTH     * sizeof(int));
	cudaMalloc(&inNeuCooData_GPU, inNeuCooSize* sizeof(int));
	cudaMalloc(&inNeuCooRow_GPU , inNeuCooSize* sizeof(int));
	cudaMalloc(&inNeuCooCol_GPU , inNeuCooSize* sizeof(int));

	cudaMalloc(&tem_out_GPU     , FMSIZE*FMSIZE*FILTNUM*sizeof(int));
	cudaMalloc(&out_GPU         , FMSIZE/3*FMSIZE/3*FILTNUM*sizeof(int));

	cudaMemcpy(filtCooNNZ_GPU , filtCooNNZ , FILTNUM*FMDEPTH *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(filtCooData_GPU, filtCooData, filtCooSize     *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(filtCooRow_GPU , filtCooRow , filtCooSize     *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(filtCooCol_GPU , filtCooCol , filtCooSize     *sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(inNeuCooNNZ_GPU , inNeuCooNNZ , FMDEPTH      *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(inNeuCooData_GPU, inNeuCooData, inNeuCooSize *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(inNeuCooRow_GPU , inNeuCooRow , inNeuCooSize *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(inNeuCooCol_GPU , inNeuCooCol , inNeuCooSize *sizeof(int), cudaMemcpyHostToDevice);

}

/***	Implement your CUDA Kernel here	***/
__global__
void convLayerGPU(int *filtCooNNZ_dev,int *filtCooData_dev,int *filtCooRow_dev,int *filtCooCol_dev,int *inNeuCooNNZ_dev,int *inNeuCooData_dev,int *inNeuCooRow_dev,int *inNeuCooCol_dev,int *tem_out_dev,int *out_dev){

	// declarations for bunch of indexing parameters
	int threadperblock = 8;
	int MaxPool = 3, MaxStride = 3;
	int fmArea = FMSIZE* FMSIZE;
	int tmpVol = fmArea* FILTNUM;
	int outArea= fmArea/(MaxPool*MaxPool);
	int outVol = outArea* FILTNUM;

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	int FmSizeAccu, FiltSizeAccu;
	int FmSizeAccu_p, FiltSizeAccu_p;
	int FmSize, FiltSize;

	int blockNum   = i/threadperblock;
	int FiltNumIdx = blockNum/FMDEPTH;
	int FmDepthIdx = blockNum%FMDEPTH;

	for (int Idx = i%(threadperblock*FMDEPTH); Idx < fmArea; Idx += threadperblock*FMDEPTH){
		int index = Idx + FiltNumIdx* fmArea;
		tem_out_dev[index] = 0;	
	}
	__syncthreads();

	if (blockNum < FMDEPTH* FILTNUM){
		FmSizeAccu   = inNeuCooNNZ_dev[FmDepthIdx];	
		if (FmDepthIdx == 0)
			FmSizeAccu_p = 0;
		else
			FmSizeAccu_p = inNeuCooNNZ_dev[FmDepthIdx-1];
	        FmSize = FmSizeAccu - FmSizeAccu_p;
        
		FiltSizeAccu = filtCooNNZ_dev[blockNum];
		if (blockNum == 0)
			FiltSizeAccu_p = 0;
		else
			FiltSizeAccu_p = filtCooNNZ_dev[blockNum-1];	
	        FiltSize = FiltSizeAccu - FiltSizeAccu_p;
        	
		for (int Idx = i%threadperblock; Idx < FmSize*FiltSize; Idx += threadperblock){
			int NeuIdx, NeuRow, NeuCol;
			int FiltIdx, FiltRow, FiltCol;
			int OutRow, OutCol, OutDepth;
			NeuIdx  = Idx%FmSize + FmSizeAccu_p;
	        	FiltIdx = Idx/FmSize + FiltSizeAccu_p;
        	
			NeuRow  = inNeuCooRow_dev[NeuIdx];
			NeuCol  = inNeuCooCol_dev[NeuIdx];
			FiltRow = filtCooRow_dev[FiltIdx];
	        	FiltCol = filtCooCol_dev[FiltIdx];
        	
			OutDepth = FiltNumIdx;
			OutRow   = NeuRow + (2 - FiltRow); 
	        	OutCol   = NeuCol + (2 - FiltCol);
        	
			if (OutRow < 0 || OutCol < 0 || OutRow >= FMSIZE || OutCol >= FMSIZE)
	        		continue;
        	
			int tmp = filtCooData_dev[FiltIdx] *inNeuCooData_dev[NeuIdx];	
			int index = OutDepth* fmArea + OutRow* FMSIZE + OutCol;
			atomicAdd(&tem_out_dev[index], tmp);
	      	}	
	}
	__syncthreads();

	int max, tmpVal;	
	for(int sli = i%FILTNUM; sli < FILTNUM; sli += FILTNUM){
		for(int fmy = 0; fmy < FMSIZE/3 ; fmy += 1){
			for(int fmx = 0; fmx < FMSIZE/3 ; fmx += 1){
				int outNeuIdx = sli*fmArea + fmy*3*FMSIZE + fmx*3;
				max = tem_out_dev[outNeuIdx];
				for(int y = 0; y < 3; y++){
					for(int x = 0; x < 3; x++){
						int ofmy = fmy*3 + y;
						int ofmx = fmx*3 + x;
						outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
						tmpVal = tem_out_dev[outNeuIdx];	
						if(tmpVal > max)
							max = tmpVal;
					}
				}
				int outIdx = sli*outArea + fmy*FMSIZE/3 + fmx;
				out_dev[outIdx] = max;
			}
		}
	}
	__syncthreads();
}
/***	Implement your CUDA Kernel here	***/

int main()
{
	//variables setting and loading input data
	timespec time_begin, time_end; 
	int convLayerCPUExecTime, convLayerGPUExecTime;

	init();
 	initCoo();

	//Convolution by CPU                                                
	clock_gettime(CLOCK_REALTIME, &time_begin);
	convLayerCPU();
	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "CPU time for executing a typical convolutional layer = "  <<  ((float)convLayerCPUExecTime)/1000 << "ms" << endl;
  
	//Convolution by GPU   

	init_Mem_GPU();
	clock_gettime(CLOCK_REALTIME, &time_begin);
	/***	Lunch your CUDA Kernel here	***/
	convLayerGPU<<<FILTNUM, 8*FMDEPTH>>>(filtCooNNZ_GPU,filtCooData_GPU,filtCooRow_GPU,filtCooCol_GPU,inNeuCooNNZ_GPU,inNeuCooData_GPU,inNeuCooRow_GPU,inNeuCooCol_GPU,tem_out_GPU,out_GPU); // Lunch the kernel

	cudaDeviceSynchronize(); // Do synchronization before clock_gettime()

	/***	Lunch your CUDA Kernel here	***/
	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = "  << ((float)convLayerGPUExecTime)/1000 << "ms" << endl;
	cudaMemcpy(outGPU, out_GPU, FILTNUM * FMSIZE/3 * FMSIZE/3*sizeof(int), cudaMemcpyDeviceToHost);	

	//check the anser from CPU and from GPU
	if(checker()){
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}
	else
		cout << "Sorry! Your result is wrong." << endl;

	/******** Added ********/
	cudaFree(&filtCooNNZ_GPU );
	cudaFree(&filtCooData_GPU);
	cudaFree(&filtCooRow_GPU );
	cudaFree(&filtCooCol_GPU );
	cudaFree(&inNeuCooNNZ_GPU );
	cudaFree(&inNeuCooData_GPU);
	cudaFree(&inNeuCooRow_GPU );
	cudaFree(&inNeuCooCol_GPU );
	cudaFree(&tem_out_GPU );
	cudaFree(&out_GPU     );
	/******** Added ********/

	//release memory space
	ending();
	
	return 0;
}
