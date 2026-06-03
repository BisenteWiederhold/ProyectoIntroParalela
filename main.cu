#include <iostream>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <ctime>

// defino la estructura base para almacenar los datos de cada particula
struct Particula {
    float x, y;
    int id_celda;
    int colisiones; 
};

// calculo las colisiones con fuerza bruta en cpu para registrar el limite del hardware
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

// calculo el spatial hashing secuencial en cpu para aislar la variable de la arquitectura
void spatialHashingCPU(Particula* particulas, int N, float tamano_celda, int ancho_grilla) {
    for (int i = 0; i < N; i++) {
        int celda_x = (int)(particulas[i].x / tamano_celda);
        int celda_y = (int)(particulas[i].y / tamano_celda);
        particulas[i].id_celda = celda_y * ancho_grilla + celda_x;
    }
}

// asigno las coordenadas a la grilla lineal de la gpu de forma paralela
__global__ void mapearAGrillaCUDA(Particula* particulas, int N, float tamano_celda, int ancho_grilla) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        int celda_x = (int)(particulas[i].x / tamano_celda);
        int celda_y = (int)(particulas[i].y / tamano_celda);
        
        particulas[i].id_celda = celda_y * ancho_grilla + celda_x;
    }
}

int main() {
    srand(time(NULL));
    
    int N = 10000; 
    size_t size = N * sizeof(Particula);
    
    std::cout << "--- INICIANDO PRUEBAS DE LA PRIMERA ENTREGA ---" << std::endl;
    std::cout << "Entidades activas: " << N << std::endl;
    
    // asigno memoria y creo las posiciones aleatorias de las particulas en el host
    Particula* h_particulas = (Particula*)malloc(size);
    for(int i = 0; i < N; i++) {
        h_particulas[i].x = static_cast<float>(rand() % 1000); 
        h_particulas[i].y = static_cast<float>(rand() % 1000); 
        h_particulas[i].id_celda = -1;
        h_particulas[i].colisiones = 0;
    }
    
    std::cout << "\n1. Ejecutando Fuerza Bruta en CPU..." << std::endl;
    
    // mido el tiempo de la fuerza bruta secuencial
    auto start_fb = std::chrono::high_resolution_clock::now();
    colisionesCPU(h_particulas, N, 5.0f);
    auto end_fb = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> tiempo_fb = end_fb - start_fb;
    std::cout << "-> Tiempo Fuerza Bruta CPU: " << tiempo_fb.count() << " ms" << std::endl;

    std::cout << "\n2. Ejecutando Spatial Hashing Secuencial en CPU..." << std::endl;
    
    // mido el tiempo del spatial hashing secuencial
    auto start_sh_cpu = std::chrono::high_resolution_clock::now();
    spatialHashingCPU(h_particulas, N, 10.0f, 100);
    auto end_sh_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> tiempo_sh_cpu = end_sh_cpu - start_sh_cpu;
    std::cout << "-> Tiempo Spatial Hashing CPU: " << tiempo_sh_cpu.count() << " ms" << std::endl;

    std::cout << "\n3. Ejecutando Spatial Hashing en GPU..." << std::endl;
    
    // reservo memoria en el device para trabajar con cuda
    Particula* d_particulas;
    cudaMalloc(&d_particulas, size);
    
    // creo los eventos nativos de cuda para medir tiempos con precision
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu); 
    cudaEventCreate(&stop_gpu);
    
    // mido la transferencia de datos desde el host hacia el device
    cudaEventRecord(start_gpu);
    cudaMemcpy(d_particulas, h_particulas, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    float latencia_ms = 0;
    cudaEventElapsedTime(&latencia_ms, start_gpu, stop_gpu);
    std::cout << "-> Latencia PCIe (Transferencia): " << latencia_ms << " ms" << std::endl;
    
    // calculo la distribucion de hilos y bloques para lanzar el kernel
    int hilos_por_bloque = 256;
    int bloques = (N + hilos_por_bloque - 1) / hilos_por_bloque;
    
    // mido la ejecucion del kernel en la gpu
    cudaEventRecord(start_gpu);
    mapearAGrillaCUDA<<<bloques, hilos_por_bloque>>>(d_particulas, N, 10.0f, 100);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    float tiempo_kernel = 0;
    cudaEventElapsedTime(&tiempo_kernel, start_gpu, stop_gpu);
    std::cout << "-> Tiempo Kernel CUDA: " << tiempo_kernel << " ms" << std::endl;
    
    // libero los recursos y destruyo los eventos creados
    cudaFree(d_particulas);
    free(h_particulas);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    
    std::cout << "\n RESULTADOS DEFINITIVOS " << std::endl;
    std::cout << "Speedup real de la arquitectura (CPU SH vs GPU SH): " << (tiempo_sh_cpu.count() / tiempo_kernel) << "x" << std::endl;
    std::cout << "Impacto teorico frente a Fuerza Bruta CPU: " << (tiempo_fb.count() / tiempo_kernel) << "x" << std::endl;
    
    return 0;
}