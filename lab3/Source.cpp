#include "mpi.h"

#include <cstdlib>
#include <iostream>
#include "time.h"

using namespace std;

// MPI routines
int ProcNum, ProcRank;

// Function to flip a matrix B of given size
void Flip(double* B, int size) {
    double temp = 0.0;
    for (int i = 0; i < size; i++) {
        for (int j = i; j < size; j++) {
            temp = B[i * size + j];
            B[i * size + j] = B[j * size + i];
            B[j * size + i] = temp;
        }
    }
}

// Matrix multiplication function
void MatrixMultiplicationMPI(double* A, double* B, double* C, int size) {
    int dim = size;
    int i, j, k, p, ind;
    double temp;
    MPI_Status Status;
    int ProcPartSize = dim / ProcNum;
    int ProcPartElem = ProcPartSize * dim;
    double* bufA = new double[ProcPartElem];
    double* bufB = new double[ProcPartElem];
    double* bufC = new double[ProcPartElem];
    int ProcPart = dim / ProcNum, part = ProcPart * dim;
    if (ProcRank == 0) {
        // Possibly some code to flip the matrix or other initializations
        cout << ProcNum << endl;
    }
    MPI_Scatter(A, part, MPI_DOUBLE, bufA, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(C, part, MPI_DOUBLE, bufC, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    temp = 0.0;
    for (i = 0; i < ProcPartSize; i++) {
        for (j = 0; j < ProcPartSize; j++) {
            for (k = 0; k < dim; k++) {
                temp += bufA[i * dim + k] * bufB[j * dim + k];
            }
            bufC[i * dim + j + ProcPartSize * ProcRank] = temp;
            temp = 0.0;
        }
    }

    int NextProc; int PrevProc;
    for (p = 1; p < ProcNum; p++) {
        NextProc = ProcRank + 1;
        if (ProcRank == ProcNum - 1)
            NextProc = 0;
        PrevProc = ProcRank - 1;
        if (ProcRank == 0)
            PrevProc = ProcNum - 1;
        MPI_Sendrecv_replace(bufB, part, MPI_DOUBLE, NextProc, 0, PrevProc, 0, MPI_COMM_WORLD, &Status);

        temp = 0.0;
        for (i = 0; i < ProcPartSize; i++) {
            for (j = 0; j < ProcPartSize; j++) {
                for (k = 0; k < dim; k++) {
                    temp += bufA[i * dim + k] * bufB[j * dim + k];
                }
                if (ProcRank - p >= 0) ind = ProcRank - p;
                else ind = (ProcNum - p + ProcRank);
                bufC[i * dim + j + ind * ProcPartSize] = temp;
                temp = 0.0;
            }
        }
    }
    MPI_Gather(bufC, ProcPartElem, MPI_DOUBLE, C, ProcPartElem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[]bufA;
    delete[]bufB;
    delete[]bufC;
    MPI_Finalize();

}
template <typename T> int matrixOutput(T* Mat, int size, string name)
{
    cout << "name matrixOutput" << endl;
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            cout << Mat[i * size + j] << " ";
        }cout << endl;
    }return 1;
}

int main(int argc, char* argv[])
{
    clock_t start;
    if (argc != 2)
    {
        cout << "error input value";
    }
    const int N = atoi(argv[1]);

    cout << "initialization" << endl;

    double* A = new double[N * N], * B = new double[N * N];
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            A[i * N + j] = (i + 1) * (j + 1);
            B[i * N + j] = (i + 1) + (2 * j + 1);
        }
    }double* C = new double[N * N];
    cout << "calc" << endl;
    start << clock();
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MatrixMultiplicationMPI(A, B, C, N);
    cout << endl << "time: " << double(clock() - start) / CLOCKS_PER_SEC << " sec" << endl;
    delete[]A;
    delete[]B;
    delete[]C;
    return 0;

}