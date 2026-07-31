

#include <stdio.h>
#include <omp.h>

#define N 3000
double a[N][N], b[N][N], c[N][N];
int main()
{
	int i, j, k;
	double t1, t2;

	// Initialization
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			a[i][j] = b[i][j] = i * j;

	t1 = omp_get_wtime();

// Multiply matrices
#pragma omp parallel for shared(a, b, c) private(i, j, k)
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			c[i][j] = 0.0;
			for (k = 0; k < N; k++)
			 c[i][j] += a[i][k] * b[k][j];

			// substitude by this line to reduce the cache miss
			// 	c[i][j] += a[i][k]*b[j][k];
			
		}
	}

	t2 = omp_get_wtime();
	printf("Time = %f\n", t2 - t1);
}
