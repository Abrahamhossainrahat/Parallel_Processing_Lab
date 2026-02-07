#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to print a matrix
void display(int rows, int cols, int matrix[rows][cols]) {
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {                                
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int K = 23, M = 3, N = 3, P = 2;

    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int A[K][M][N], B[K][N][P], R[K][M][P];

    if(rank == 0) {
        for(int k = 0; k < K; k++)
            for(int i = 0; i < M; i++)
                for(int j = 0; j < N; j++)
                    A[k][i][j] = rand() % 5;

        for(int k = 0; k < K; k++)
            for(int i = 0; i < N; i++)
                for(int j = 0; j < P; j++)
                    B[k][i][j] = rand() % 5;
    }

    // Calculate dynamic start and end index for each process
    int base = K / size;
    int rem  = K % size;

    int start = rank * base + (rank < rem ? rank : rem);
    int end   = start + base + (rank < rem ? 1 : 0);

    int localK = end - start;  // number of matrices for this process

    // Allocate local matrices dynamically
    int localA[localK][M][N];
    int localB[localK][N][P];
    int localR[localK][M][P];

    // Scatter manually since sizes vary
    if(rank == 0) {
        for(int r = 0; r < size; r++) {
            int r_start = r * base + (r < rem ? r : rem);
            int r_end   = r_start + base + (r < rem ? 1 : 0);
            int rK = r_end - r_start;

            if(r == 0) { // copy to own local buffer
                for(int k = 0; k < rK; k++)
                    for(int i = 0; i < M; i++)
                        for(int j = 0; j < N; j++)
                            localA[k][i][j] = A[r_start + k][i][j];

                for(int k = 0; k < rK; k++)
                    for(int i = 0; i < N; i++)
                        for(int j = 0; j < P; j++)
                            localB[k][i][j] = B[r_start + k][i][j];
            } else { // send to other process
                MPI_Send(&A[r_start][0][0], rK*M*N, MPI_INT, r, 0, MPI_COMM_WORLD);
                MPI_Send(&B[r_start][0][0], rK*N*P, MPI_INT, r, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(&localA[0][0][0], localK*M*N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&localB[0][0][0], localK*N*P, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // Matrix multiplication
    for(int k = 0; k < localK; k++) {
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < P; j++) {
                localR[k][i][j] = 0;
                for(int l = 0; l < N; l++) {
                    localR[k][i][j] += (localA[k][i][l] * localB[k][l][j]) % 100;
                }
                localR[k][i][j] %= 100;
            }
        }
    }

    double endTime = MPI_Wtime();

    // Gather results manually
    if(rank == 0) {
        // copy own result
        for(int k = 0; k < localK; k++)
            for(int i = 0; i < M; i++)
                for(int j = 0; j < P; j++)
                    R[start + k][i][j] = localR[k][i][j];

        for(int r = 1; r < size; r++) {
            int r_start = r * base + (r < rem ? r : rem);
            int r_end   = r_start + base + (r < rem ? 1 : 0);
            int rK = r_end - r_start;

            MPI_Recv(&R[r_start][0][0], rK*M*P, MPI_INT, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(&localR[0][0][0], localK*M*P, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    //Print all the result matrices
    if(rank == 0) {
        for(int k = 0; k < K; k++) {
            printf("Result Matrix R%d\n", k);
            display(M, N, A[k]);
            display(N, P, B[k]);
            display(M, P, R[k]);
        }
    }

    // Print timing
    //printf("Process %d: Time taken = %f seconds\n", rank, endTime - startTime);

    MPI_Finalize();
    return 0;
}

/* Nessary code
 sudo apt install openmpi-bin openmpi-common libopenmpi-dev
 To compile : mpicc -o matrix_mul_dynamic  matrix_mul_dynamic.c 
 To run : mpirun -np 4 ./matrix_mul_dynamic 2>\dev\null >output1.txt
*/

