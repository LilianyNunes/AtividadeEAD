#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

__global__ void blellochUpSweep(int* d_array, int size, int step) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && (idx + 1) % (2 * step) == 0) {
        d_array[idx] += d_array[idx - step];
    }
}

__global__ void blellochDownSweep(int* d_array, int size, int step) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && (idx + 1) % (2 * step) == 0) {
        int temp = d_array[idx - step];
        d_array[idx - step] = d_array[idx];
        d_array[idx] += temp;
    }
}

void runBlelloch(int size) {
    int* h_input = new int[size];
    for (int i = 0; i < size; i++) {
        h_input[i] = 1; // Inicializa para somar, cada elemento com valor 1
    }

    int* d_array;
    cudaMalloc(&d_array, size * sizeof(int));

    // Copia os dados de entrada para o dispositivo
    cudaMemcpy(d_array, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    int steps = static_cast<int>(std::log2(size)) + 1; // Contagem de passos
    int work = 0; // Inicializa o contador de trabalho

    // Marca o início do tempo
    auto start = std::chrono::high_resolution_clock::now();

    // Fase Up-Sweep (Construção da soma prefixada)
    for (int step = 1; step < size; step <<= 1) {
        blellochUpSweep << <gridSize, blockSize >> > (d_array, size, step);
        cudaDeviceSynchronize();
        work += size / (2 * step);  // Cada etapa reduz a quantidade de operações realizadas
    }

    // Define o último elemento para zero antes de começar a fase Down-Sweep
    cudaMemset(d_array + size - 1, 0, sizeof(int));

    // Fase Down-Sweep (Distribuição da soma prefixada)
    for (int step = size / 2; step >= 1; step >>= 1) {
        blellochDownSweep << <gridSize, blockSize >> > (d_array, size, step);
        cudaDeviceSynchronize();
        work += size / (2 * step);  // A quantidade de trabalho por etapa decresce
    }

    // Marca o fim do tempo
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> exec_time = end - start;

    // Copia o resultado final de volta para o host
    cudaMemcpy(h_input, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Exibe o resultado, tempo de execução, quantidade de trabalho e número de etapas
    std::cout << "Tamanho do Array: " << size << " -> Soma Final: " << h_input[size - 1]
        << " -> Tempo de Execucao: " << exec_time.count() << " segundos"
        << " -> Trabalho Realizado: " << work
        << " -> Etapas Necessarias: " << steps << std::endl;

    // Libera memória
    cudaFree(d_array);
    delete[] h_input;
}

int main() {
    int sizes[] = { 100, 1000, 10000, 100000, 1000000, 10000000 };
    for (int size : sizes) {
        runBlelloch(size);
    }
    return 0;
}
