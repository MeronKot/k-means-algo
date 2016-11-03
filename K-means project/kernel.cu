#include "k-means_header.h"

void error(Point *dev_points, Point *dev_clusters, float *dev_dis)
{
	cudaFree(dev_points);
	cudaFree(dev_clusters);
	cudaFree(dev_dis);
}

__global__ void distanceKernel(Point *points,Point *clusterPoints,float *dis, int n,int procId,int k)
{
    float x = 0,y = 0;
	int p = threadIdx.x;
	
	if( (p >= (procId * k)) && (p < (procId * k) + k) )
	{
		p = p % k;
		for(int i = 0 ; i < n ; i++)
		{
			x = clusterPoints[p].x - points[i].x;
			y = clusterPoints[p].y - points[i].y;
			dis[p*n + i] = sqrt(x*x + y*y);
		}
	}
}

cudaError_t distanceWithCuda(Point *points, Point *clusterPoints,  float **dis , int n, int k,int procId,int numprocs)
{
    Point *dev_points = 0;
    Point *dev_clusters = 0;
	float *dev_dis = 0;
	cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        error(dev_points,dev_clusters,dev_dis);
    }
    // Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_points, n * sizeof(Point));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        error(dev_points,dev_clusters,dev_dis);
    }

	cudaStatus = cudaMalloc((void**)&dev_clusters, k * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        error(dev_points,dev_clusters,dev_dis);
    }
	
	cudaStatus = cudaMalloc((void**)&dev_dis, k * n * sizeof(float));

	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        error(dev_points,dev_clusters,dev_dis);
    }

    // Copy input vectors from host memory to GPU buffers.

    cudaStatus = cudaMemcpy(dev_points, points, n * sizeof(Point), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        error(dev_points,dev_clusters,dev_dis);
    }

	cudaStatus = cudaMemcpy(dev_clusters, clusterPoints, k * sizeof(Point), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		error(dev_points,dev_clusters,dev_dis);
    }

	//cudaStatus = cudaMemcpy2D(dev_dis,pitch,dis,pitch,k,n,cudaMemcpyHostToDevice);
	for(int i = 0 ; i < k ; i++)
		cudaStatus = cudaMemcpy(&dev_dis[i * n], dis[i],n * sizeof(float), cudaMemcpyHostToDevice);
	
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        error(dev_points,dev_clusters,dev_dis);
	}

    // Launch a kernel on the GPU with one thread for each element.
	distanceKernel<<<1, k*numprocs>>>(dev_points,dev_clusters,dev_dis,n,procId - 1,k);
	  
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "distanceKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        error(dev_points,dev_clusters,dev_dis);
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching distanceKernel!\n", cudaStatus);
         error(dev_points,dev_clusters,dev_dis);
    }

    // Copy output vector from GPU buffer to host memory.
    for (int i = 0 ; i < k ; i++)
		cudaStatus = cudaMemcpy(dis[i], &dev_dis[i * n], n * sizeof(float), cudaMemcpyDeviceToHost);
	
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        error(dev_points,dev_clusters,dev_dis);
    }
   
	cudaFree(dev_points);
	cudaFree(dev_clusters);
	cudaFree(dev_dis);
	return cudaStatus;
}


