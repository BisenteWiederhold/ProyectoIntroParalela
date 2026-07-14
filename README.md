# Benchmark de colisiones: CPU vs GPU

Este proyecto sirve para comparar distintos métodos de detección de colisiones 2D (estilo bullet hell) usando C++ y CUDA. 

El programa mide el rendimiento dividiendo el trabajo en dos partes:
- Broad phase: la búsqueda general de posibles choques.
- Narrow phase: el cálculo matemático exacto entre dos partículas.

## Requisitos

* Nvidia CUDA toolkit instalado.
* Raylib (las carpetas include y lib deben estar junto al código fuente).
* Herramientas de compilación de C++ de visual studio.

## Como compilar

En Windows, es necesario usar la consola especial de visual studio para que encuentre el compilador de C++.

1. Abre el menú inicio y busca "x64 Native Tools Command Prompt".
2. Navega con la consola hasta la carpeta de tu proyecto.
3. Copia y pega esta línea para compilar:

```bat
nvcc bh.cu -o bh.exe -O3 -I raylib/include -L raylib/lib -lraylib -lwinmm -lgdi32 -luser32 -lshell32 -Xcompiler /MD
```
### Flags

* -o bh.exe: define el nombre del archivo ejecutable que se va a crear.
* -O3: aplica el nivel máximo de optimización. Es clave usarlo para que el código de CPU corra a su máxima velocidad y la comparativa sea justa.
* -I y -L: le indican al compilador en qué carpetas buscar los archivos de raylib.
* -lraylib, -lwinmm, -lgdi32, -luser32, -lshell32: enlazan raylib y las herramientas internas de windows que se necesitan para dibujar la ventana y manejar las entradas del teclado.
* -Xcompiler /MD: le dice al compilador de visual studio que use la versión correcta de las librerías del sistema para evitar conflictos de memoria.

## Uso del programa

Al abrir el archivo bh.exe, el programa no mostrará la ventana gráfica de inmediato. Primero hará una prueba de rendimiento en el fondo probando desde 10,000 hasta 100,000 partículas.
Cuando termine la prueba de fondo, se abrirá la ventana de la simulación.

### Controles

* Espacio: cambia el método de colisión (fuerza bruta, sweep & prune, spatial hashing CPU, spatial hashing GPU).
* N: suma 10,000 partículas.
* M: resta 10,000 partículas.

## Resultados

Al terminar la prueba inicial, el programa creará un archivo llamado benchmark_fases.csv. Este archivo tiene los tiempos medidos en milisegundos de cada prueba.
