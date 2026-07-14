#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include "raylib.h"

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "Error CUDA: " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } }

#define ANCHO_MAPA 1280
#define ALTO_MAPA 1020
#define TAM_CELDA 20.0f
#define COLS ((int)(ANCHO_MAPA / TAM_CELDA) + 1)
#define ROWS ((int)(ALTO_MAPA / TAM_CELDA) + 1)
#define TOTAL_CELDAS (COLS * ROWS)
#define MAX_VECINOS_POR_CELDA 50
#define MAX_PARTICULAS 100000 

struct Entidad {
    int id;
    float x, y, vx, vy, radio; 
    int is_colliding;
};

// Estructura para separar los tiempos
struct Tiempos {
    float broad;
    float narrow;
    float total() const { return broad + narrow; }
};

// Kernels CUDA
__global__ void buildSpatialGrid(Entidad* entidades, int num_entidades, int* grid_counters, int* grid_cells, int max_por_celda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_entidades) return;
    Entidad e = entidades[i];
    int min_col = max(0, (int)((e.x - e.radio) / TAM_CELDA));
    int max_col = min(COLS - 1, (int)((e.x + e.radio) / TAM_CELDA));
    int min_row = max(0, (int)((e.y - e.radio) / TAM_CELDA));
    int max_row = min(ROWS - 1, (int)((e.y + e.radio) / TAM_CELDA));
    for (int r = min_row; r <= max_row; r++) {
        for (int c = min_col; c <= max_col; c++) {
            int hash_id = r * COLS + c;
            int pos = atomicAdd(&grid_counters[hash_id], 1);
            if (pos < max_por_celda) grid_cells[hash_id * max_por_celda + pos] = e.id;
        }
    }
}

__global__ void checkCollisions(Entidad* entidades, int num_entidades, int* grid_counters, int* grid_cells, int max_por_celda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_entidades) return;
    Entidad my_e = entidades[i];
    int col = (int)(my_e.x / TAM_CELDA);
    int row = (int)(my_e.y / TAM_CELDA);
    int hash_id = row * COLS + col;
    int vecinos_en_celda = min(grid_counters[hash_id], max_por_celda);
    bool hit = false;
    for (int k = 0; k < vecinos_en_celda; k++) {
        int vecino_id = grid_cells[hash_id * max_por_celda + k];
        if (vecino_id != my_e.id) {
            Entidad v = entidades[vecino_id];
            float dx = my_e.x - v.x;
            float dy = my_e.y - v.y;
            float dist_sq = dx*dx + dy*dy;
            float rad_sum = my_e.radio + v.radio;
            if (dist_sq <= (rad_sum * rad_sum)) { hit = true; break; }
        }
    }
    entidades[i].is_colliding = hit ? 1 : 0;
}

// Funciones CPU
Tiempos checkCollisionsBruteForceCPU(std::vector<Entidad>& entidades, int n) {
    Tiempos t;
    t.broad = 0.0f; // La fuerza bruta no tiene broad phase
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        bool hit = false;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                float dx = entidades[i].x - entidades[j].x;
                float dy = entidades[i].y - entidades[j].y;
                float rad_sum = entidades[i].radio + entidades[j].radio;
                if ((dx*dx + dy*dy) <= (rad_sum * rad_sum)) { hit = true; break; }
            }
        }
        entidades[i].is_colliding = hit ? 1 : 0;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    t.narrow = std::chrono::duration<float, std::milli>(t2 - t1).count();
    return t;
}

Tiempos checkCollisionsSweepAndPruneCPU(std::vector<Entidad>& entidades, int n) {
    Tiempos t;
    
    // Broad phase: ordenar por el eje x
    auto t1 = std::chrono::high_resolution_clock::now();
    std::sort(entidades.begin(), entidades.begin() + n, [](const Entidad& a, const Entidad& b) {
        return (a.x - a.radio) < (b.x - b.radio);
    });
    auto t2 = std::chrono::high_resolution_clock::now();
    t.broad = std::chrono::duration<float, std::milli>(t2 - t1).count();

    // Narrow phase: recorrer y hacer check de distancias
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        float max_x_i = entidades[i].x + entidades[i].radio;
        bool hit = false;
        for (int j = i + 1; j < n; j++) {
            if ((entidades[j].x - entidades[j].radio) > max_x_i) break; 
            float dx = entidades[i].x - entidades[j].x;
            float dy = entidades[i].y - entidades[j].y;
            float rad_sum = entidades[i].radio + entidades[j].radio;
            if ((dx*dx + dy*dy) <= (rad_sum * rad_sum)) { hit = true; break; }
        }
        entidades[i].is_colliding = hit ? 1 : 0;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    t.narrow = std::chrono::duration<float, std::milli>(t4 - t3).count();
    
    return t;
}

Tiempos checkCollisionsSpatialHashingCPU(std::vector<Entidad>& entidades, std::vector<int>& grid_counters, std::vector<int>& grid_cells, int n) {
    Tiempos t;
    
    // Broad phase: llenar la grilla espacial
    auto t1 = std::chrono::high_resolution_clock::now();
    std::fill(grid_counters.begin(), grid_counters.end(), 0);
    for (int i = 0; i < n; i++) {
        Entidad& e = entidades[i];
        int min_col = std::max(0, (int)((e.x - e.radio) / TAM_CELDA));
        int max_col = std::min(COLS - 1, (int)((e.x + e.radio) / TAM_CELDA));
        int min_row = std::max(0, (int)((e.y - e.radio) / TAM_CELDA));
        int max_row = std::min(ROWS - 1, (int)((e.y + e.radio) / TAM_CELDA));
        for (int r = min_row; r <= max_row; r++) {
            for (int c = min_col; c <= max_col; c++) {
                int hash_id = r * COLS + c;
                int pos = grid_counters[hash_id]++; 
                if (pos < MAX_VECINOS_POR_CELDA) grid_cells[hash_id * MAX_VECINOS_POR_CELDA + pos] = e.id;
            }
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    t.broad = std::chrono::duration<float, std::milli>(t2 - t1).count();

    // Narrow phase: buscar colisiones en la misma celda
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        Entidad& e = entidades[i];
        int col = (int)(e.x / TAM_CELDA);
        int row = (int)(e.y / TAM_CELDA);
        int hash_id = row * COLS + col;
        int vecinos = std::min(grid_counters[hash_id], MAX_VECINOS_POR_CELDA);
        bool hit = false;
        for (int k = 0; k < vecinos; k++) {
            int vec_id = grid_cells[hash_id * MAX_VECINOS_POR_CELDA + k];
            if (vec_id != e.id) {
                Entidad& v = entidades[vec_id];
                float dx = e.x - v.x;
                float dy = e.y - v.y;
                float rad_sum = e.radio + v.radio;
                if ((dx*dx + dy*dy) <= (rad_sum * rad_sum)) { hit = true; break; }
            }
        }
        e.is_colliding = hit ? 1 : 0;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    t.narrow = std::chrono::duration<float, std::milli>(t4 - t3).count();
    
    return t;
}

int main() {
    std::vector<Entidad> h_entidades(MAX_PARTICULAS);
    for (int i = 0; i < MAX_PARTICULAS; i++) {
        h_entidades[i].id = i;
        h_entidades[i].x = static_cast<float>(rand() % ANCHO_MAPA);
        h_entidades[i].y = static_cast<float>(rand() % ALTO_MAPA);
        h_entidades[i].vx = static_cast<float>((rand() % 400) - 200) / 100.0f;
        h_entidades[i].vy = static_cast<float>((rand() % 400) - 200) / 100.0f;
        h_entidades[i].radio = (rand() % 100 < 5) ? 8.0f : 3.0f; 
        h_entidades[i].is_colliding = 0;
    }

    std::vector<int> cpu_grid_counters(TOTAL_CELDAS, 0);
    std::vector<int> cpu_grid_cells(TOTAL_CELDAS * MAX_VECINOS_POR_CELDA, 0);

    Entidad* d_entidades;
    int *d_grid_counters, *d_grid_cells;
    CHECK_CUDA(cudaMalloc(&d_entidades, MAX_PARTICULAS * sizeof(Entidad)));
    CHECK_CUDA(cudaMalloc(&d_grid_counters, TOTAL_CELDAS * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_grid_cells, TOTAL_CELDAS * MAX_VECINOS_POR_CELDA * sizeof(int)));

    // Eventos CUDA para broad y narrow separados
    cudaEvent_t start_gpu, mid_gpu, stop_gpu;
    cudaEventCreate(&start_gpu); cudaEventCreate(&mid_gpu); cudaEventCreate(&stop_gpu);
    
    // Benchmark automatizado para los gráficos
    std::vector<int> test_sizes = {10000, 20000, 40000, 50000, 70000, 90000, 100000};
    std::ofstream archivo("benchmark_fases.csv");
    
    // Nueva cabecera con fases separadas
    archivo << "N_Particulas,BF_Narrow,SAP_Broad,SAP_Narrow,SH_CPU_Broad,SH_CPU_Narrow,SH_GPU_Broad,SH_GPU_Narrow\n";
    std::cout << "Iniciando benchmark separado por fases (puede tardar un poco en N grandes...)" << std::endl;

    for (int test_N : test_sizes) {
        std::cout << "Testeando N = " << test_N << "..." << std::endl;

        // 1. Fuerza bruta
        Tiempos t_bf = checkCollisionsBruteForceCPU(h_entidades, test_N);

        // 2. Sweep & prune
        Tiempos t_sap = checkCollisionsSweepAndPruneCPU(h_entidades, test_N);

        // 3. Spatial hashing CPU
        Tiempos t_sh_cpu = checkCollisionsSpatialHashingCPU(h_entidades, cpu_grid_counters, cpu_grid_cells, test_N);

        // 4. Spatial hashing GPU
        int blockSize = 256;
        int gridSize = (test_N + blockSize - 1) / blockSize;
        
        // Memcpy fuera de los contadores de algoritmo
        CHECK_CUDA(cudaMemcpyAsync(d_entidades, h_entidades.data(), test_N * sizeof(Entidad), cudaMemcpyHostToDevice));
        
        // Broad phase GPU: memset y build grid
        cudaEventRecord(start_gpu);
        CHECK_CUDA(cudaMemsetAsync(d_grid_counters, 0, TOTAL_CELDAS * sizeof(int)));
        buildSpatialGrid<<<gridSize, blockSize>>>(d_entidades, test_N, d_grid_counters, d_grid_cells, MAX_VECINOS_POR_CELDA);
        
        // Narrow phase GPU: comprobación de colisiones
        cudaEventRecord(mid_gpu);
        checkCollisions<<<gridSize, blockSize>>>(d_entidades, test_N, d_grid_counters, d_grid_cells, MAX_VECINOS_POR_CELDA);
        cudaEventRecord(stop_gpu);
        cudaEventSynchronize(stop_gpu);

        float ms_gpu_broad = 0, ms_gpu_narrow = 0;
        cudaEventElapsedTime(&ms_gpu_broad, start_gpu, mid_gpu);
        cudaEventElapsedTime(&ms_gpu_narrow, mid_gpu, stop_gpu);

        CHECK_CUDA(cudaMemcpyAsync(h_entidades.data(), d_entidades, test_N * sizeof(Entidad), cudaMemcpyDeviceToHost));

        // Guardar fila en el CSV
        archivo << test_N << "," 
                << t_bf.narrow << "," 
                << t_sap.broad << "," << t_sap.narrow << "," 
                << t_sh_cpu.broad << "," << t_sh_cpu.narrow << "," 
                << ms_gpu_broad << "," << ms_gpu_narrow << "\n";
    }
    archivo.close();
    std::cout << "Benchmark completado, resultados en 'benchmark_fases.csv'" << std::endl;

    int N = 10000; 
    InitWindow(ANCHO_MAPA, ALTO_MAPA, "Bullet Hell Benchmark - Analisis de Fases");
    SetTargetFPS(60);

    int metodo = 3;
    std::string nombres[] = {"BruteForce", "Sweep&Prune", "SpatialHashing CPU", "SpatialHashing GPU"};

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) metodo = (metodo + 1) % 4;
        if (IsKeyPressed(KEY_N) && N <= 90000) N += 10000;
        if (IsKeyPressed(KEY_M) && N > 10000) N -= 10000;

        for (int i = 0; i < N; i++) {
            h_entidades[i].x += h_entidades[i].vx;
            h_entidades[i].y += h_entidades[i].vy;
            if (h_entidades[i].x < 0 || h_entidades[i].x > ANCHO_MAPA) h_entidades[i].vx *= -1;
            if (h_entidades[i].y < 0 || h_entidades[i].y > ALTO_MAPA) h_entidades[i].vy *= -1;
            h_entidades[i].is_colliding = 0;
        }

        Tiempos t_actual = {0, 0};

        if (metodo == 0) {
            t_actual = checkCollisionsBruteForceCPU(h_entidades, N);
        } else if (metodo == 1) {
            t_actual = checkCollisionsSweepAndPruneCPU(h_entidades, N);
        } else if (metodo == 2) {
            t_actual = checkCollisionsSpatialHashingCPU(h_entidades, cpu_grid_counters, cpu_grid_cells, N);
        } else {
            int gSize = (N + 255) / 256;
            CHECK_CUDA(cudaMemcpyAsync(d_entidades, h_entidades.data(), N * sizeof(Entidad), cudaMemcpyHostToDevice));
            
            cudaEventRecord(start_gpu);
            CHECK_CUDA(cudaMemsetAsync(d_grid_counters, 0, TOTAL_CELDAS * sizeof(int)));
            buildSpatialGrid<<<gSize, 256>>>(d_entidades, N, d_grid_counters, d_grid_cells, MAX_VECINOS_POR_CELDA);
            cudaEventRecord(mid_gpu);
            checkCollisions<<<gSize, 256>>>(d_entidades, N, d_grid_counters, d_grid_cells, MAX_VECINOS_POR_CELDA);
            cudaEventRecord(stop_gpu);
            cudaEventSynchronize(stop_gpu);

            cudaEventElapsedTime(&t_actual.broad, start_gpu, mid_gpu);
            cudaEventElapsedTime(&t_actual.narrow, mid_gpu, stop_gpu);
            
            CHECK_CUDA(cudaMemcpyAsync(h_entidades.data(), d_entidades, N * sizeof(Entidad), cudaMemcpyDeviceToHost));
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);
        for (int i = 0; i < N; i++) DrawRectangle(h_entidades[i].x, h_entidades[i].y, h_entidades[i].radio * 2, h_entidades[i].radio * 2, h_entidades[i].is_colliding ? MAROON : SKYBLUE);
        DrawRectangle(10, 10, 320, 130, Fade(BLACK, 0.8f));
        DrawText(TextFormat("N: %i | FPS: %i", N, GetFPS()), 20, 20, 20, GREEN);
        DrawText(nombres[metodo].c_str(), 20, 45, 20, GOLD);
        
        // Renderizado de las fases por separado
        DrawText(TextFormat("Broad:  %.3f ms", t_actual.broad), 20, 75, 18, ORANGE);
        DrawText(TextFormat("Narrow: %.3f ms", t_actual.narrow), 20, 95, 18, RED);
        DrawText(TextFormat("TOTAL:  %.3f ms", t_actual.total()), 20, 115, 18, WHITE);
        
        EndDrawing();
    }
    
    cudaEventDestroy(start_gpu); cudaEventDestroy(mid_gpu); cudaEventDestroy(stop_gpu);
    cudaFree(d_entidades); cudaFree(d_grid_counters); cudaFree(d_grid_cells);
    CloseWindow();
    return 0;
}