#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

__global__ void hillisSteeleKernel(int* d_input, int* d_output, int size, int step) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    if (idx >= step) {
        d_output[idx] = d_input[idx] + d_input[idx - step];
    }
    else {
        d_output[idx] = d_input[idx];
    }
}

void runHillisSteele(int size) {
    // Inicializa o array de entrada com valores
    int* h_input = new int[size];
    for (int i = 0; i < size; i++) {
        h_input[i] = 1; // Pode ajustar o valor conforme necessário
    }

    int* d_input, * d_output;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));

    // Copia dados para o dispositivo
    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    // Configuração do Kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    int steps = static_cast<int>(std::log2(size)) + 1;  // Número de etapas (passos)
    int work = 0;  // Quantidade de trabalho realizada

    // Registra o tempo de início
    auto start = std::chrono::high_resolution_clock::now();

    // Executa o algoritmo de Hillis-Steele
    for (int step = 1; step < size; step <<= 1) {
        hillisSteeleKernel << <gridSize, blockSize >> > (d_input, d_output, size, step);
        cudaDeviceSynchronize();
        std::swap(d_input, d_output);
        work += size;  // Cada passo realiza `size` operações
    }

    // Registra o tempo de fim
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> exec_time = end - start;

    // Copia o resultado final para o host
    cudaMemcpy(h_input, d_input, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Exibe o resultado, tempo de execução, quantidade de trabalho e número de etapas
    std::cout << "Tamanho do Array: " << size << " -> Soma Final: " << h_input[size - 1]
        << " -> Tempo de Execucao: " << exec_time.count() << " segundos"
        << " -> Trabalho Realizado: " << work
        << " -> Etapas Necessarias: " << steps << std::endl;

    // Libera memória
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
}

int main() {
    int sizes[] = { 100, 1000, 10000, 100000, 1000000, 10000000 };
    for (int size : sizes) {
        runHillisSteele(size);
    }
    return 0;
}
