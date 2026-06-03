#include <iostream>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <iomanip>

// 1. ESTRUCTURA BASE
struct Particula {
    float x, y;
    int id_celda;
    int colisiones; 
};

// 2. LÍNEA BASE CPU (Fuerza Bruta)
void colisionesCPU(Particula* particulas, int N, float radio_colision) {
    for (int i = 0; i < N; i++) {
        particulas[i].colisiones = 0;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                float dx = particulas[i].x - particulas[j].x;
                float dy = particulas[i].y - particulas[j].y;
                float distancia = sqrt(dx*dx + dy*dy);
                
                if (distancia < radio_colision) {
                    particulas[i].colisiones++;
                }
            }
        }
    }
}

// 3. KERNEL CUDA (Mapeo a Grilla Spatial Hashing)
__global__ void mapearAGrillaCUDA(Particula* particulas, int N, float tamano_celda, int ancho_grilla) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        int celda_x = (int)(particulas[i].x / tamano_celda);
        int celda_y = (int)(particulas[i].y / tamano_celda);
        
        particulas[i].id_celda = celda_y * ancho_grilla + celda_x;
    }
}

// 4. FUNCIÓN PRINCIPAL (Automatizada para múltiples cargas)
int main() {
    srand(time(NULL));
    
    // Lista de tamaños de prueba
    std::vector<int> pruebas_N = {10000, 50000, 100000};
    
    std::cout << "==========================================\n";
    std::cout << "INICIANDO BENCHMARK ESCALONADO (ENTREGA 1)\n";
    std::cout << "==========================================\n\n";

    // Variables para guardar los resultados y hacer la tabla final
    std::vector<float> tiempos_cpu;
    std::vector<float> tiempos_gpu;
    std::vector<float> latencias;

    for (int N : pruebas_N) {
        std::cout << "-> Evaluando con N = " << N << " particulas...\n";
        size_t size = N * sizeof(Particula);
        
        Particula* h_particulas = (Particula*)malloc(size);
        for(int i = 0; i < N; i++) {
            h_particulas[i].x = static_cast<float>(rand() % 1000);
            h_particulas[i].y = static_cast<float>(rand() % 1000);
            h_particulas[i].id_celda = -1;
            h_particulas[i].colisiones = 0;
        }
        
        // --- LÍNEA BASE CPU ---
        auto start_cpu = std::chrono::high_resolution_clock::now();
        colisionesCPU(h_particulas, N, 5.0f);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> t_cpu = end_cpu - start_cpu;
        tiempos_cpu.push_back(t_cpu.count());

        // --- CUDA Y PCIe ---
        Particula* d_particulas;
        cudaMalloc(&d_particulas, size);
        
        cudaEvent_t start_gpu, stop_gpu;
        cudaEventCreate(&start_gpu); 
        cudaEventCreate(&stop_gpu);
        
        // Medir latencia HtoD
        cudaEventRecord(start_gpu);
        cudaMemcpy(d_particulas, h_particulas, size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop_gpu);
        cudaEventSynchronize(stop_gpu);
        
        float latencia_ms = 0;
        cudaEventElapsedTime(&latencia_ms, start_gpu, stop_gpu);
        latencias.push_back(latencia_ms);
        
        // Lanzar Kernel
        int hilos_por_bloque = 256;
        int bloques = (N + hilos_por_bloque - 1) / hilos_por_bloque;
        
        cudaEventRecord(start_gpu);
        mapearAGrillaCUDA<<<bloques, hilos_por_bloque>>>(d_particulas, N, 10.0f, 100);
        cudaEventRecord(stop_gpu);
        cudaEventSynchronize(stop_gpu);
        
        float t_kernel = 0;
        cudaEventElapsedTime(&t_kernel, start_gpu, stop_gpu);
        tiempos_gpu.push_back(t_kernel);
        
        cudaFree(d_particulas);
        free(h_particulas);
        cudaEventDestroy(start_gpu);
        cudaEventDestroy(stop_gpu);
        
        std::cout << "   [OK] Prueba superada.\n\n";
    }
    
    // --- IMPRIMIR TABLA RESUMEN PARA LA DIAPOSITIVA ---
    std::cout << "===============================================================\n";
    std::cout << "TABLA DE RESULTADOS FINALES (PARA PRESENTACION)\n";
    std::cout << "===============================================================\n";
    std::cout << std::left << std::setw(15) << "Particulas (N)" 
              << std::setw(15) << "T. CPU (ms)" 
              << std::setw(15) << "Latencia PCIe" 
              << std::setw(15) << "T. GPU (ms)" 
              << std::setw(15) << "Speedup" << "\n";
    std::cout << "---------------------------------------------------------------\n";
    
    for (size_t i = 0; i < pruebas_N.size(); i++) {
        float speedup = tiempos_cpu[i] / tiempos_gpu[i];
        std::cout << std::left << std::setw(15) << pruebas_N[i] 
                  << std::setw(15) << tiempos_cpu[i] 
                  << std::setw(15) << latencias[i] 
                  << std::setw(15) << tiempos_gpu[i] 
                  << speedup << "x\n";
    }
    std::cout << "===============================================================\n";

    return 0;
}