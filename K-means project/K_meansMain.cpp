#include "k-means_header.h"

int main (int argc, char *argv[])
{
	Point *points;
	Point *slavePartOfPoints; 
	Point *clusterPoints;
	Point **clusterReceieved;
	float dt,interval;
	float **distances;
	int done = FALSE;
	int n,k,l,z,limit,position;
	int rank,numprocs;
	int blocklen[7] = {1,1,1,1,1,1,1};
	
	FILE *file;
	FILE *output;
	MPI_Datatype POINT;
	MPI_Datatype type[7] = { MPI_INT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_INT };
	MPI_Aint disp[7];
	MPI_Status status;
	float initpara[5];	
	
	//arrays that contain cluster no. for each point.
	int *startClusters;
	int *endClusters;
    int *slaveClusters;
	
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(numprocs < 2)
	{
		printf("Num of processes need to be greater then 2");
		MPI_Abort(MPI_COMM_WORLD,0);
	}

	//MPI struct creation
	for (int idx = 0; idx < sizeof(Point) / sizeof(int); idx++)
		disp[idx] = idx * sizeof(int);

	MPI_Type_create_struct(sizeof(Point) / sizeof(int),blocklen,disp,type,&POINT);
	MPI_Type_commit(&POINT);
		
//############## Master Only ########################
	if(rank == 0)
	{	
		//File reading...
		file = fopen("pointsCheck2.txt","r");
		readFromFile(file,&n,&k,&dt,&interval,&limit);
		checkInput(n,k,dt,interval,limit,numprocs);
		points = (Point*) malloc (n * sizeof(Point));
		if (points == NULL)
		{
			printf("Error! memory not allocated points");
			exit(0);
		}
		readPoints(file,points,n);
		fclose(file);
		
		//Split all the points into n - 1 processors because master is not participate, 
		//para array for communication between processors.
		position = n / (numprocs - 1);

		initpara[0] = (float)position;
		initpara[1] = dt;
		initpara[2] = interval;
		initpara[3] = (float)k;
		initpara[4] = (float)limit;
		

		for (int i = 1 ; i < numprocs  ; i++)
			MPI_Send(&initpara[0],5,MPI_FLOAT,i,0,MPI_COMM_WORLD);
		for(int i = 1 ; i < numprocs ; i++)
			MPI_Send(points + (i - 1) * position,position,POINT,i,0,MPI_COMM_WORLD);		
		
		//Array initializations and allocations
		startClusters = (int*)malloc(n * sizeof(int));
		endClusters = (int*)malloc(n * sizeof(int));
		
		clusterPoints = (Point*) malloc (k * sizeof(Point));
		//Array of cluster points for saving all the iterations and than find the minimum
		clusterReceieved = (Point**)malloc((int(interval / dt)) * sizeof(Point*));
		for(int i = 0 ; i < int(interval / dt); i++)
			clusterReceieved[i] = (Point*)malloc(k * sizeof(Point));
	
		if((startClusters == NULL) || (endClusters == NULL) || (clusterPoints == NULL) || (clusterReceieved == NULL))                 
		{
			printf("Error! memory not allocated");
			exit(0);
		}
		

		//The first clusters will be the first k points.	
		for(int i = 0 ; i < k ; i++)
		{
			clusterPoints[i].x = points[i].a + points[i].r;
			clusterPoints[i].y = points[i].b;
		}
			
		double start = MPI_Wtime();//Start measuring time.
		
		z = 0;//idx for the array of clusters.
		//Run k-means algo on every position of the points.
		for (float m = 0 ; m < interval ; m += dt )
		{
			for(int index = 0 ; index < n; index++)//Position calculation
			{
				points[index].x = points[index].a + (points[index].r * cos((2*M_PI*m)/interval));
				points[index].y = points[index].b + (points[index].r * sin((2*M_PI*m)/interval));
			}
			for(int i = 0 ; i < k ; i++)
			{
				clusterPoints[i].x = points[i].x;
				clusterPoints[i].y = points[i].y;
			}
			
			for(int i = 1 ; i < numprocs ; i++)
				MPI_Send(clusterPoints,k,POINT,i,0,MPI_COMM_WORLD);
		
			initArray(startClusters,n);
			initArray(endClusters,n);

			l=0;//idx for max iterations of k-means on specific position
			while (1)
			{				
				//Receiving from all the procs their part of points to clusters job
				for(int i = 1 ; i < numprocs ; i++)
					MPI_Recv(endClusters+(position * (i - 1)),position,MPI_INT,i,0,MPI_COMM_WORLD,&status);
							
				//Calculate new clusters after move
				calculateNewClusters(points,endClusters,clusterPoints,k,n);

				//Check if there is any change or below the limit
				if((isDiffer(endClusters,startClusters,n) == 0) || (l > limit))
					done = TRUE;
				else
					for(int i = 0 ; i < n ; i++)
						startClusters[i] = endClusters[i];
				
				//Send all procs if job done
				for(int i = 1 ; i < numprocs ; i++ )
					MPI_Send(&done,1,MPI_INT,i,0,MPI_COMM_WORLD);

				//Finish algo.
				if(done == TRUE)
					break;
				
				
				//Continue algo again
				for(int i = 1 ; i < numprocs ; i++ )
					MPI_Send(clusterPoints,k,POINT,i,0,MPI_COMM_WORLD);

				l++;
			}
			//Recording iteration result for future calculation.
			memcpy(clusterReceieved[z],clusterPoints,k * sizeof(Point));
			done = FALSE;
			z++;
		}
		//Calculate the minimum distance between all iterations and get the idx into minCluster.
		int minCluster = findMinimumResult(clusterReceieved,int(interval/dt),k);

		double end = MPI_Wtime();//Stop time.
		
		//Printing result details...
		output = fopen("output.txt","w");
		fprintf(output,"d = %f\nt = %f\nCenters:\n",calculateDisBetweenCentroids(clusterReceieved[minCluster],k),(float)minCluster * dt);
		for(int i = 0 ; i < k ; i++)
			fprintf(output,"%f   %f\n",clusterReceieved[minCluster][i].x,clusterReceieved[minCluster][i].y);
		fprintf(output,"\ntime = %f",end - start);
		fclose(output);
		printf("Done");

		//Free allocations
		free(startClusters);
		free(endClusters);
		free(clusterPoints);
		for(int i = 0 ; i < int(interval / dt); i++)
			free(clusterReceieved[i]);
		free(clusterReceieved);
	}
//############## Slave Only ########################	
	else
	{
		//Receiving all required arguments from master
		MPI_Recv(&initpara,5,MPI_FLOAT,0,0,MPI_COMM_WORLD,&status);
		
		//Allocations..
		slavePartOfPoints = (Point*) malloc ((int)(initpara[0]) * sizeof(Point));
		distances = (float**) malloc ((int)initpara[3] * sizeof(float*));
		clusterPoints = (Point*) malloc ((int)initpara[3] * sizeof(Point));
		//Array the contains the clusters of the part of points received.
		slaveClusters = (int*)malloc((int)initpara[0] * sizeof(int));
		
		if ((slavePartOfPoints == NULL) || (distances == NULL) || (clusterPoints == NULL) || (slaveClusters == NULL))
		{
			printf("Error! memory not allocated.");
			exit(0);
		}

		for(int i = 0 ; i < (int)initpara[3] ; i++)
		{
			distances[i] = (float*) malloc ((int)initpara[0] * sizeof(float));
			for(int j = 0 ; j < (int)initpara[0] ; j++)
				distances[i][j] = 0.0;
		}
		
		//Receiving from master part of points for calculation and the cluster points
		MPI_Recv(slavePartOfPoints,(int)initpara[0],POINT,0,0,MPI_COMM_WORLD,&status);
			
		//Find distances from the points to clusters using cuda and decide the cluster for each point
		//on every position of the points.
		for(float m = 0 ; m < initpara[2]; m += initpara[1])
		{			
			//Position calculation
			for(int index = 0 ; index < (int)initpara[0]; index++)
			{
				slavePartOfPoints[index].x = slavePartOfPoints[index].a + (slavePartOfPoints[index].r * cos((2*M_PI*m)/initpara[2]));
				slavePartOfPoints[index].y = slavePartOfPoints[index].b + (slavePartOfPoints[index].r * sin((2*M_PI*m)/initpara[2]));
			}
			
			MPI_Recv(clusterPoints,(int)initpara[3],POINT,0,0,MPI_COMM_WORLD,&status);
			
			initArray(slaveClusters,(int)initpara[0]);						
			//Calculate cluster to each point and send to master until it stops.
			while(1)
			{
				//distanceWithCuda(slavePartOfPoints,clusterPoints,distances,(int)initpara[0],(int)initpara[3],rank,numprocs - 1);
				
				distance(slavePartOfPoints,clusterPoints,distances,(int)initpara[0],(int)initpara[3],rank,numprocs - 1);

				chooseClusterToEachPoint(slavePartOfPoints,clusterPoints,slaveClusters,distances,(int)initpara[0],(int)initpara[3]);

				MPI_Send(slaveClusters,(int)initpara[0],MPI_INT,0,0,MPI_COMM_WORLD);

				initArray(slaveClusters,(int)initpara[0]);						

				MPI_Recv(&done,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);

				if(done == TRUE)
					break;

				MPI_Recv(clusterPoints,(int)initpara[3],POINT,0,0,MPI_COMM_WORLD,&status);
			}//End calculation
		}//End iterations
		
		//Free allocations...
		free(slavePartOfPoints);
		for(int i = 0 ; i < (int)initpara[3]; i++)
			free(distances[i]);
		free(distances);
		free(clusterPoints);
		free(slaveClusters);
	}//End slave
	
	MPI_Finalize();
	return 0;
}
