#include "k-means_header.h"


void readFromFile(FILE *file, int *n, int *k, float *dt, float *interval, int *limit)
{
	fscanf(file,"%d %d %f %f %d\n",n,k,dt,interval,limit);
}

void readPoints(FILE *file, Point *points, int n)
{
	int id;
	float a,b,r;
	for(int i = 0 ; i < n ; i++)
	{
		fscanf(file,"%d %f %f %f\n",&id,&a,&b,&r);
		points[i].id = id;
		points[i].a = a;
		points[i].b = b;
		points[i].r = r;
	}
}
/*
	int n,k,limit,j;
	float dt,T;
	int id,offset,size = 60;
	float a,b,r;
	char *buffer;
	MPI_Status status;
	MPI_Offset fsize;
	
	MPI_File_get_size(*file,&fsize);
	
	buffer = (char*)malloc(fsize);
	MPI_File_seek(*file,0,MPI_SEEK_SET);
	MPI_File_read(*file,buffer,fsize,MPI_CHAR,&status);
	sscanf(buffer,"%d %d %f %f %d%n",&n,&k,&dt,&T,&limit,&offset);
	buffer += offset;
	for(int i = 0 ; i < n ; i++)
	{
		sscanf(buffer,"%d %f %f %f%n",&id,&a,&b,&r,&offset);
		points[i].id = id;
		points[i].a = a;
		points[i].b = b;
		points[i].r = r;
		buffer += offset;
	}
	*/