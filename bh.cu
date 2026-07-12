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

// Spatial hashing
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

// Colisiones
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

// Fuerza bruta CPU
void checkCollisionsBruteForceCPU(std::vector<Entidad>& entidades, int n) {
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
}

// Sweep and prune CPU
void checkCollisionsSweepAndPruneCPU(std::vector<Entidad>& entidades, int n) {
    std::sort(entidades.begin(), entidades.begin() + n, [](const Entidad& a, const Entidad& b) {
        return (a.x - a.radio) < (b.x - b.radio);
    });
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
}

// Spatial hashing CPU
void checkCollisionsSpatialHashingCPU(std::vector<Entidad>& entidades, std::vector<int>& grid_counters, std::vector<int>& grid_cells, int n) {
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
}

int main() {
    int N = 10000; 

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

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu); cudaEventCreate(&stop_gpu);
    
    // Benchmark a txt
    std::ofstream archivo("benchmark_inicial.txt");
    auto t1 = std::chrono::high_resolution_clock::now();
    checkCollisionsBruteForceCPU(h_entidades, N);
    auto t2 = std::chrono::high_resolution_clock::now();
    archivo << "BruteForce: " << std::chrono::duration<float, std::milli>(t2 - t1).count() << "ms\n";
    t1 = std::chrono::high_resolution_clock::now();
    checkCollisionsSweepAndPruneCPU(h_entidades, N);
    t2 = std::chrono::high_resolution_clock::now();
    archivo << "Sweep&Prune: " << std::chrono::duration<float, std::milli>(t2 - t1).count() << "ms\n";
    t1 = std::chrono::high_resolution_clock::now();
    checkCollisionsSpatialHashingCPU(h_entidades, cpu_grid_counters, cpu_grid_cells, N);
    t2 = std::chrono::high_resolution_clock::now();
    archivo << "SpatialHashing CPU: " << std::chrono::duration<float, std::milli>(t2 - t1).count() << "ms\n";
    
    float ms_gpu = 0;
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    cudaEventRecord(start_gpu);
    CHECK_CUDA(cudaMemcpyAsync(d_entidades, h_entidades.data(), N * sizeof(Entidad), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemsetAsync(d_grid_counters, 0, TOTAL_CELDAS * sizeof(int)));
    buildSpatialGrid<<<gridSize, blockSize>>>(d_entidades, N, d_grid_counters, d_grid_cells, MAX_VECINOS_POR_CELDA);
    checkCollisions<<<gridSize, blockSize>>>(d_entidades, N, d_grid_counters, d_grid_cells, MAX_VECINOS_POR_CELDA);
    CHECK_CUDA(cudaMemcpyAsync(h_entidades.data(), d_entidades, N * sizeof(Entidad), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&ms_gpu, start_gpu, stop_gpu);
    archivo << "SpatialHashing GPU: " << ms_gpu << "ms\n";
    archivo.close();

    InitWindow(ANCHO_MAPA, ALTO_MAPA, "Bullet Hell");
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

        float t_ms = 0;
        if (metodo == 0) {
            auto tr1 = std::chrono::high_resolution_clock::now();
            checkCollisionsBruteForceCPU(h_entidades, N);
            t_ms = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - tr1).count();
        } else if (metodo == 1) {
            auto tr1 = std::chrono::high_resolution_clock::now();
            checkCollisionsSweepAndPruneCPU(h_entidades, N);
            t_ms = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - tr1).count();
        } else if (metodo == 2) {
            auto tr1 = std::chrono::high_resolution_clock::now();
            checkCollisionsSpatialHashingCPU(h_entidades, cpu_grid_counters, cpu_grid_cells, N);
            t_ms = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - tr1).count();
        } else {
            int gSize = (N + 255) / 256;
            cudaEventRecord(start_gpu);
            CHECK_CUDA(cudaMemcpyAsync(d_entidades, h_entidades.data(), N * sizeof(Entidad), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemsetAsync(d_grid_counters, 0, TOTAL_CELDAS * sizeof(int)));
            buildSpatialGrid<<<gSize, 256>>>(d_entidades, N, d_grid_counters, d_grid_cells, MAX_VECINOS_POR_CELDA);
            checkCollisions<<<gSize, 256>>>(d_entidades, N, d_grid_counters, d_grid_cells, MAX_VECINOS_POR_CELDA);
            CHECK_CUDA(cudaMemcpyAsync(h_entidades.data(), d_entidades, N * sizeof(Entidad), cudaMemcpyDeviceToHost));
            cudaEventRecord(stop_gpu);
            cudaEventSynchronize(stop_gpu);
            cudaEventElapsedTime(&t_ms, start_gpu, stop_gpu);
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);
        for (int i = 0; i < N; i++) DrawRectangle(h_entidades[i].x, h_entidades[i].y, h_entidades[i].radio * 2, h_entidades[i].radio * 2, h_entidades[i].is_colliding ? MAROON : SKYBLUE);
        DrawRectangle(10, 10, 300, 100, Fade(BLACK, 0.7f));
        DrawText(TextFormat("N: %i | FPS: %i", N, GetFPS()), 20, 20, 20, GREEN);
        DrawText(nombres[metodo].c_str(), 20, 45, 20, GOLD);
        DrawText(TextFormat("Calc: %.3f ms", t_ms), 20, 70, 20, WHITE);
        EndDrawing();
    }
    cudaFree(d_entidades); cudaFree(d_grid_counters); cudaFree(d_grid_cells);
    CloseWindow();
    return 0;
} 