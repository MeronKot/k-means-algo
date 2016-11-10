#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include <memory.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define FALSE 0
#define TRUE 1

struct Point
{
	int id;
	float a;
	float b;
	float x;
	float y;
	float r;
	int cluster;
};


void error(Point *dev_points, Point *dev_clusters, float **dev_dis);

void checkInput(int n,int k,float dt,float interval,int limit,int numprocs);
cudaError_t distanceWithCuda(Point *points, Point *clusterPoints,  float **dis , int n, int k,int procId,int numprocs);

void chooseClusterToEachPoint(Point *points, Point *clusters,int* slaveClusters,float** distances,int n,int k);

float calculateDisBetweenCentroids(Point *clusters,int k);

void readFromFile(FILE *file, int *n, int *k, float *dt, float *interval, int *limit);

void readPoints(FILE *file, Point *points,int n);

void initArray(int *points,int numOfPoints);

void calculateNewClusters(Point *points,int *endClusters,Point *clusters,int k,int n);

int isDiffer(int *endClusters, int *startClusters,int n);

int findMinimum(float *disBetweenCentroids, int size);

int findMinimumResult(Point **clusters,int intervals,int k);

float distanceBetweenPoints(Point x,Point y);

void distance(Point *points,Point *clusters,float **distances,float n,float k,int rank,int numprocs);