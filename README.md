# NCTU 2017 Computer Architecture Final Project - Part 2

**Part-2：Implement sparse convolution , relu , and maxpooling in convLayerGPU() with CUDA and store your result in the outGPU and use NVVP to analyze your code.**

## Download
At command line type:
<pre>
git clone https://github.com/s0606757/CA2017FP-Part2.git
</pre>

## Three sub-directory

### ./data
This directory contains the input data for the base program：
* . /data/filter_COO.txt - Store the values of sparse filters
* . /data/neuron_COO.txt - Store the values of sparse neurons

### ./device
The program under this directory can show the device information.
#### usage
<pre>
cd ./device
make
make run
</pre>

### ./example
There are two examples(InnerProduct and Matrix Multiplication) under this directory.
#### usage
<pre>
cd ./example/InnerProduct/
make
make run

or

cd ./example/MatrixMultiplication/
make
make run
</pre>

## Usage of the base program
<pre>
make
make run
</pre>

## Evaluation
We will compare the execution time to get the speedup by
<pre>
Speedup = convLayerCPU_execTime / convLayerGPU_execTime
PS：convLayerGPU_execTime includes : cudaMalloc time 、cudaMemcpy time and execution time. 
Please  put these functions between clock_gettime(CLOCK_REALTIME, &time_begin) between clock_gettime(CLOCK_REALTIME, &time_end)
</pre>

## Grading Policy
**(A) Completeness (35%)**<br/>
&nbsp;    Your result(convLayerGPU) must be correct (Pass the check) (10%)<br/>
&nbsp;&nbsp;&nbsp;    Your design(convLayerGPU) is faster than convLayerCPU() (15%)<br/>
&nbsp;&nbsp;&nbsp;   Use your data format(10%) <br/>
**(B) Report (35%)**<br/>
&nbsp;&nbsp;&nbsp;    Describe your implementation algorithm and explain your results (15%)<br/>
&nbsp;&nbsp;&nbsp;    Discuss what kind of optimization you did( it is better or worse?) (10%)<br/>
&nbsp;&nbsp;&nbsp;    Show how you use NVVP to help you find and solve perf. issues (5%)<br/>
&nbsp;&nbsp;&nbsp;    Feedback of this part (5%)<br/>
**(C) Performance Rank (30%)**<br/>
&nbsp;&nbsp;&nbsp;    We will rank your CUDA kernels’ performance(execution time) on GTX K20C<br/>
&nbsp;&nbsp;&nbsp;    The fastest one will get 30% and the last one will get 1%<br/>

## Other Rules
* It’s a team work, 1 ~ 4 people in one team <br/>
   - Register (**Please reedit!**)[here](https://docs.google.com/spreadsheets/d/1cSKtAuxRwu-y8bwys1Dka8hY58U_clqU0l4W81Cis3Y/edit?usp=sharing) before deadline.<br/>
* [Account list](https://docs.google.com/spreadsheets/d/1hLfJjv58QsXRwLlma45IflcpicqlQFgYiKp77vlJokk/edit#gid=0)
* Compress your **code** and **report** into one zip file and upload to E3.<br/>
* Name your report as：**LeaderID_Report_FP2.pdf**<br/>
* Name your package as： **LeaderID_FP2.zip**<br/>
* One team only need to upload **one** package to E3.<br/>
* Make sure TA can compile and run your code with “make” and “make run” on the provided server.<br/>
* **Any CUDA library is forbidden to use in this project !!!** <br/>
* **DELAY IS NOT ACCEPTABLE !!!** <br/>
* **Due day：2017/11/30(Thr) 23:50** <br/>

## Useful Reference
* Neural Network Pruning [Here](https://arxiv.org/pdf/1506.02626.pdf)
* Sparsity in Neurons [Here](http://www.ece.ubc.ca/~aamodt/papers/Cnvlutin.ISCA2016.pdf)
* Sparse Foramat in Matrix Multiplication [Here](https://pdfs.semanticscholar.org/9abb/086fabdcd2853ed8303c0f9a62cf4b917a62.pdf)
* Implement Sparse Matrix Multiplication with CUDA [Here](http://wnbell.com/media/2008-12-NVR-SpMV/nvr-2008-004.pdf)
* An Accelerator for Sparse CNN [Here](http://people.csail.mit.edu/anurag_m/papers/2017.scnn.isca.pdf)
* Sparse Format - Nvidia [Here](https://drive.google.com/file/d/0B-mvsV4UBCFFbEhpMzFIbUVLVGs/view?usp=sharing )
* Cuda-Debuging Command Line (It is helpful for you) [Here](https://drive.google.com/file/d/0B-mvsV4UBCFFZ24yeG96TEFVa0k/view?usp=sharing)


# 2017CA_FP2
