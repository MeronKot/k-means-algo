#include "k-means_header.h"


void chooseClusterToEachPoint(Point *points, Point *clusters,int* slaveClusters,float** distances,int n,int k)
{
	int i,j;
	int cluster = 0;
	float min;

	//Run over all the points and decide which cluster is closer to every point.
#pragma parallel omp for private(i)
{
	for(i = 0 ; i < n; i++)
	{
		min = distances[0][i];
		cluster = 0;
		for(j = 0 ; j < k; j++)
			if(distances[j][i] < min)
			{
				min = distances[j][i];
				cluster = j;
			}
		slaveClusters[i] = cluster;
	}
}
}

float calculateDisBetweenCentroids(Point *clusters,int k)
{
	int i,j;
	float x,y;
	float mindis;

	mindis = distanceBetweenPoints(clusters[0],clusters[1]);
	for(i = 0 ; i < k ; i++)
		for(j = 0 ; j < k ; j++)
		{
			if(i != j)
			{
				float temp;
				x = clusters[i].x - clusters[j].x;
				y = clusters[i].y - clusters[j].y;
				temp = sqrt(x*x + y*y);
				if(temp < mindis)
					mindis = temp;
			}
		}
	return mindis;
}
	/*
	//Run over all the clusters and sum the distances between the points of the clusters.
	for (i = 0 ; i < k ; i++)
	{
		if((i + 1) < k)
		{
			x = clusters[i].x - clusters[i+1].x;
			y = clusters[i].y - clusters[i+1].y;
			dis += sqrt(x*x + y*y);
		}
		else
		{
			x = clusters[i].x - clusters[0].x;
			y = clusters[i].y - clusters[0].y;
			dis += sqrt(x*x + y*y);
		}
	}
	return dis;
	*/
int findMinimum(float *disBetweenCentroids, int size)
{
	int i = 0;
	int minIdx = 0;
	float min = 0;
	//Run over all the distances between centroids and decide which is the minimum
	min = disBetweenCentroids[i];
	for (i = 1 ; i < size ; i++)
		if(disBetweenCentroids[i] < min)
		{
			min = disBetweenCentroids[i];
			minIdx = i;
		}
	//Return the idx for future calculations.
	return minIdx;
}

void initArray(int *points,int numOfPoints)
{
	int i;
#pragma parallel omp for private(i)
	{
		for(i = 0 ; i < numOfPoints ; i++)
			points[i] = -1;
	}
}

void calculateNewClusters(Point *points,int *endClusters,Point *clusters,int k,int n)
{
	int i;
	int *count = (int*)malloc(k * sizeof(int));
	Point *newClusters = (Point*)malloc(k * sizeof(Point));

	if((count == NULL) || (newClusters == NULL))
		printf("Error! memory not allocated");
	
	//Init all the arrays
	for (i = 0 ; i < k ; i++)
	{
		count[i] = 0;
		newClusters[i].x = 0;
		newClusters[i].y = 0;
	}
	//Run over all the points and count the number of points each cluster have and sum x and y.
#pragma parallel omp for private(i)
{
	for(i = 0 ; i < n ; i++)
	{
		count[endClusters[i]]++;
		newClusters[endClusters[i]].x += points[i].x;
		newClusters[endClusters[i]].y += points[i].y;
	}
}
	//Divid the x and y of each cluster by the num of point each cluster have to calculate central mass
	for(i = 0 ; i < k ; i++)
	{
		if(count[i] != 0)
		{
			newClusters[i].x /= count[i];
			newClusters[i].y /= count[i];
		}
		else
		{
			newClusters[i].x = clusters[i].x;
			newClusters[i].y = clusters[i].y;
		}
	}
	//Return the result to given clusters
	for(i = 0 ; i < k ; i++)
	{
		clusters[i].x = newClusters[i].x;
		clusters[i].y = newClusters[i].y;
	}
	free(count);
	free(newClusters);
}

int isDiffer(int *endClusters, int *startClusters,int n)
{
	int i;
#pragma parallel omp for private(i)
{
	for(i = 0 ; i < n ; i++)
		if(startClusters[i] != endClusters[i])
			return -1;
}
	return 0;
}

int findMinimumResult(Point **clusters,int intervals,int k)
{
	int i;
	int minCluster = 0;
	float *minDis;

	minDis = (float*)malloc(intervals * sizeof(float));
	if(minDis == NULL)
		printf("Error! memory not allocated");

	//Run over all the intervals and calculate for each the distance between the clusters
#pragma parallel omp for private(i)
{	
	for (i = 0 ; i < intervals ; i++)
	{
		minDis[i] = calculateDisBetweenCentroids(clusters[i],k);
	}
}
	//Find the minimum among them
	minCluster = findMinimum(minDis,intervals);
	free(minDis);
	//Return the min idx
	return minCluster;
}

float distanceBetweenPoints(Point p1,Point p2)
{
	float x,y,dis;
	
	x = p1.x - p2.x;
	y = p1.y - p2.y;

	return sqrt(x*x + y*y);
}