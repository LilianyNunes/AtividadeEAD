#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void reduceKernel(int* d_input, int* d_output, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += d_input[i];
    }
    *d_output = sum;
}

void runReduceSerial(int size) {
    // Inicializa o array de entrada com valores
    int* h_input = new int[size];
    for (int i = 0; i < size; i++) {
        h_input[i] = 1; // ou outro valor conforme necessário
    }

    int h_output = 0;
    int* d_input, * d_output;

    // Aloca memória no dispositivo
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));

    // Copia dados para o dispositivo
    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    // Registra o tempo de início
    auto start = std::chrono::high_resolution_clock::now();

    // Lança o kernel de redução serial
    reduceKernel << <1, 1 >> > (d_input, d_output, size);

    // Sincroniza a GPU e registra o tempo de fim
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> exec_time = end - start;

    // Copia o resultado de volta para o host
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Exibe o resultado e o tempo de execução
    std::cout << "Tamanho do Array: " << size << " -> Soma: " << h_output
        << " -> Tempo de Execucao: " << exec_time.count() << " segundos" << std::endl;

    // Libera memória
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
}

int main() {
    int sizes[] = { 100, 1000, 10000, 100000, 1000000, 10000000 };
    for (int size : sizes) {
        runReduceSerial(size);
    }
    return 0;
}
