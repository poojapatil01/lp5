#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Function to perform parallel reduction for minimum value
int parallel_min(const vector<int>& data) {
    int min_val = data[0];

    // Perform parallel reduction for minimum value
    #pragma omp parallel for reduction(min:min_val)
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
    }

    return min_val;
}

// Function to perform parallel reduction for maximum value
int parallel_max(const vector<int>& data) {
    int max_val = data[0];

    // Perform parallel reduction for maximum value
    #pragma omp parallel for reduction(max:max_val)
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    return max_val;
}

// Function to perform parallel reduction for sum
int parallel_sum(const vector<int>& data) {
    int sum_val = 0;

    // Perform parallel reduction for sum
    #pragma omp parallel for reduction(+:sum_val)
    for (size_t i = 0; i < data.size(); ++i) {
        sum_val += data[i];
    }

    return sum_val;
}

// Function to perform parallel reduction for average
double parallel_average(const vector<int>& data) {
    int sum_val = parallel_sum(data);

    return static_cast<double>(sum_val) / data.size();
}

int main() {
    const int data_size = 10;
    
    vector<int> data(data_size);
    cout << "Enter " << data_size << " elements for the data array:" << endl;
    for (int i = 0; i < data_size; ++i) {
        cout << "Element " << i + 1 << ": ";
        cin >> data[i];
    }

    int min_val = parallel_min(data);
    int max_val = parallel_max(data);
    int sum_val = parallel_sum(data);
    double avg_val = parallel_average(data);

    cout << "Minimum value: " << min_val << endl;
    cout << "Maximum value: " << max_val << endl;
    cout << "Sum: " << sum_val << endl;
    cout << "Average: " << avg_val << endl;

    return 0;
}
