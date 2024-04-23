#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <mpi.h>

// Function to perform partitioning
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }

    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Function to perform parallel quicksort
void parallelQuicksort(std::vector<int>& arr, int low, int high, int rank, int size) {
    if (low < high) {
        int pivotIndex = partition(arr, low, high);

        // Determine which process holds the pivot
        int pivotRank = (pivotIndex < low + high) / 2;

        // Broadcast pivotRank to all processes
        MPI_Bcast(&pivotRank, 1, MPI_INT, rank, MPI_COMM_WORLD);

        if (rank == pivotRank) {
            // Send subarray to other processes
            for (int i = 0; i < size; i++) {
                if (i != rank) {
                    MPI_Send(&low, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                    MPI_Send(&high, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                    MPI_Send(arr.data() + low, high - low + 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            // Receive subarray
            MPI_Recv(&low, 1, MPI_INT, pivotRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&high, 1, MPI_INT, pivotRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            arr.resize(high - low + 1);
            MPI_Recv(arr.data(), high - low + 1, MPI_INT, pivotRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Parallel partitioning
        pivotIndex = partition(arr, low, high);

        // Recursive parallel quicksort
        parallelQuicksort(arr, low, pivotIndex - 1, rank, size);
        parallelQuicksort(arr, pivotIndex + 1, high, rank, size);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 20;
    std::vector<int> arr(n);

    // Generate random data on root process
    if (rank == 0) {
        srand(time(NULL));
        for (int i = 0; i < n; ++i) {
            arr[i] = rand() % 100;
        }
    }

    // Broadcast the data to all processes
    MPI_Bcast(arr.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform parallel quicksort
    parallelQuicksort(arr, 0, n - 1, rank, size);

    // Gather the sorted data to root process
    std::vector<int> sortedData(n * size);
    MPI_Gather(arr.data(), n, MPI_INT, sortedData.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    // Print sorted data on root process
    if (rank == 0) {
        std::cout << "Sorted array:" << std::endl;
        for (int i = 0; i < n * size; ++i) {
            std::cout << sortedData[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}
