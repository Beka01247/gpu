# CUDA C++ для мидки: теория с нуля и шаблоны

## 0. С чего начать

Любое решение мидки — это один файл `lastname.cu` из пяти частей, которые всегда идут в одном и том же порядке. Выучи этот порядок, и пустой файл перестанет пугать.

1. **Подключения и константы**: `#include`, макрос `CHECK`, `const int THREADS = 256;`
2. **`__device__` функции**: операция над одним элементом (сложить, умножить, скалярное произведение строки на столбец).
3. **`__global__` кернелы**: заполнение массивов, вычисление, редукция (сумма).
4. **Host-функции**: то же вычисление на CPU, `verify`, печать массивов.
5. **`main`**: инфо о GPU → память → стримы → prefetch на GPU → fill → вычисление → редукция → синхронизация → prefetch на CPU → CPU-версия → verify → печать времени → освобождение памяти.

### Что подключать

Никакие внешние библиотеки не нужны, линковать ничего не надо. Хватает стандартных заголовков:

| Заголовок | Зачем |
| --- | --- |
| `#include <cstdio>` | `printf` |
| `#include <cstdlib>` | `malloc`, `free`, `exit` |
| `#include <chrono>` | замер времени на CPU |
| `#include <cuda_runtime.h>` | весь CUDA API (`nvcc` подключает его сам, но лучше написать явно) |
| `#include <climits>` | `INT_MAX`, `INT_MIN` — нужны для редукции max/min |
| `#include <cmath>` | `fabs` — если сравниваешь `float`/`double` |

### Как компилировать и запускать

```bash
vim serikov.cu
nvcc -O2 -o serikov serikov.cu
./serikov
```

Файл обязательно с расширением `.cu`: файл `.cpp` `nvcc` компилирует как обычный C++, и `<<< >>>` там не работает.

### Порядок работы на экзамене

Это рекомендованные шаги из мидки 2025, они же самый безопасный путь:

1. Скопируй каркас из раздела 12 (подключения, `CHECK`, `main` с выводом информации о GPU), скомпилируй пустым.
2. Напиши CPU-версию и проверь её на маленьком N (10 или 4×4), напечатав массивы.
3. Напиши кернел, запусти, сравни с CPU через `verify`.
4. Добавь стримы, prefetch, редукцию, таймеры.
5. Увеличь N до того, что требует задание.

Компилируй после каждого шага. Ошибку в 10 строках найти легко, ошибку в 150 строках — нет.

## 1. Модель CUDA: host и device

**Host** — это CPU и его память (RAM). **Device** — это GPU и его память (VRAM). `main` всегда выполняется на CPU и оттуда запускает работу на GPU.

### Три вида функций

| Метка | Вызывается из | Работает на | Особенности |
| --- | --- | --- | --- |
| `__global__` (кернел) | host | GPU | всегда `void`, запуск через `<<< >>>`, выполняется сразу тысячами потоков |
| `__device__` | кернела или другой `__device__` | GPU | может возвращать значение, с CPU вызвать нельзя |
| `__host__ __device__` | отовсюду | CPU и GPU | одна функция для обеих сторон |
| без метки | host | CPU | обычная C++ функция |

```cpp
__device__ int addDevice(int x, int y) { return x + y; }   // one element

__global__ void addKernel(const int *a, const int *b, int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = addDevice(a[i], b[i]);
}
```

### Запуск кернела

```cpp
kernel<<<blocks, threads, sharedBytes, stream>>>(args...);
```

- `blocks` — сколько блоков, `threads` — сколько потоков в каждом блоке.
- `sharedBytes` — динамическая shared memory; обычно `0`.
- `stream` — в каком стриме запускать; без него — дефолтный стрим. Запуск `<<<blocks, threads, 0, s1>>>` и есть «использование стримов» из грейдинга.
- Аргументы копируются по значению. Указатели должны указывать на память, доступную GPU (`cudaMalloc` или `cudaMallocManaged`), а не на обычный `malloc`.
- Кернел ничего не возвращает: результат всегда пишется в массив или в переменную в памяти GPU.

### Асинхронность — самое важное

Запуск кернела возвращается на CPU сразу, не дожидаясь GPU. Поэтому перед тем, как читать результат на CPU, обязательно:

```cpp
CHECK(cudaDeviceSynchronize());   // wait until the GPU finishes
```

Забыл синхронизацию — читаешь нули или мусор, а `verify` падает. `printf` внутри кернела работает, но вывод появится только после синхронизации.

## 2. Иерархия потоков: сколько блоков и потоков

На экзамене правило фиксировано: **blocks = warpSize × число SM, threads = 256**. Ниже объяснение, почему.

Запуск кернела создаёт **grid** (сетку) из **blocks** (блоков), а каждый блок состоит из **threads** (потоков). Все потоки выполняют один и тот же код кернела. Отличаются они только встроенными переменными:

| Переменная | Что значит | Диапазон |
| --- | --- | --- |
| `threadIdx.x` | номер потока внутри блока | 0 … blockDim.x − 1 |
| `blockIdx.x` | номер блока в сетке | 0 … gridDim.x − 1 |
| `blockDim.x` | сколько потоков в блоке (= threads из `<<< >>>`) | до 1024 |
| `gridDim.x` | сколько блоков (= blocks из `<<< >>>`) | — |

### Глобальный индекс потока

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

Пример: blockDim.x = 4. Поток 1 в блоке 2 получает idx = 2 × 4 + 1 = 9. Так каждый поток получает свой уникальный номер от 0 до blocks × threads − 1.

### Warp и SM

- **Warp** — группа из 32 потоков, которые GPU выполняет вместе, одной инструкцией. Поэтому число потоков в блоке делают кратным 32: иначе часть последнего warp простаивает.
- **SM (Streaming Multiprocessor)** — вычислительный блок GPU, их десятки. Каждый блок потоков целиком выполняется на одном SM и никогда не делится.
- Если число блоков не кратно числу SM, в последней «волне» часть SM простаивает («fallow SMs» из лекции week-5). Отсюда правило: blocks = 32 × число SM — кратно числу SM и даёт каждому SM много работы.

### Как получить число SM и warpSize

Способ 1 — через `cudaDeviceProp` (заодно даёт имя GPU для вывода «Hardware name»):

```cpp
int device;
CHECK(cudaGetDevice(&device));
cudaDeviceProp prop;
CHECK(cudaGetDeviceProperties(&prop, device));
int numSMs = prop.multiProcessorCount;
int blocks = prop.warpSize * numSMs;
printf("Hardware name: %s\n", prop.name);
```

Способ 2 — через атрибуты (так делали в курсе NVIDIA):

```cpp
int numSMs, warp;
CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
CHECK(cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, device));
int blocks = warp * numSMs;
```

### Почему threads = 256

- Кратно 32 (целое число warp), не больше лимита 1024.
- Степень двойки — это обязательно для редукции (раздел 8).
- 128–512 обычно дают лучшую загрузку SM; 256 — стандартный безопасный выбор.
- В коде объяви константой с комментарием — так закрывается пункт «Declare the optimum number of threads»: `const int THREADS = 256; // multiple of warp size, power of two`

Для матриц можно запускать 2D-сетку через `dim3` (тогда появляются `.y`), но проще и надёжнее держать 1D-сетку и пересчитывать индекс в строку и столбец (раздел 9).

## 3. Grid-stride loop

Grid-stride loop нужен в **каждом** кернеле: блоков фиксированно 32 × SM, и потоков часто меньше, чем элементов. Каждый поток берёт свой элемент и прыгает вперёд на общее число потоков в сетке.

```cpp
__global__ void kernel(int *x, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // first element
    int stride = blockDim.x * gridDim.x;               // total threads in the grid
    for (int i = idx; i < n; i += stride)
        x[i] = x[i] * 2;
}
```

Или короче, одной строкой:

```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
```

Пример: 2 блока по 4 потока = 8 потоков, n = 20. Поток с idx = 3 обрабатывает элементы 3, 11, 19. Поток с idx = 5 — элементы 5 и 13.

Почему это хорошо:

- Работает при любом N: и когда элементов больше потоков, и когда меньше (при N = 10 лишние потоки просто не входят в цикл).
- Условие `i < n` заодно защищает от выхода за границы массива.
- Соседние потоки читают соседние ячейки памяти — это быстро (coalesced access).

Частая ошибка: `stride = blockDim.x` без `* gridDim.x`. Тогда разные блоки делают одну и ту же работу, а сумма в редукции выходит в разы больше правильной.

## 4. Память: выделение, перенос, освобождение

На экзамене бери **Unified Memory + prefetch**: это меньше кода и сразу закрывает пункт «asynchronous data migration». Явные копирования — запасной вариант.

### Вариант A: Unified Memory (рекомендую)

`cudaMallocManaged` выделяет память, которую видят и CPU, и GPU, по одному и тому же указателю. Данные сами переезжают туда, где их трогают, но медленно: через page faults, по кусочкам. `cudaMemPrefetchAsync` перевозит всё заранее и целиком — это и есть «оптимальная миграция данных».

```cpp
size_t bytes = N * sizeof(int);
int *a;
CHECK(cudaMallocManaged(&a, bytes));

// before kernels: move to the GPU
CHECK(cudaMemPrefetchAsync(a, bytes, device, stream));      // stream is optional
kernel<<<blocks, THREADS, 0, stream>>>(a, N);

// before the CPU reads: move back
CHECK(cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId));
CHECK(cudaDeviceSynchronize());                            // prefetch is async too!
printf("%d\n", a[0]);

CHECK(cudaFree(a));
```

- `device` — номер GPU из `cudaGetDevice(&device)`; `cudaCpuDeviceId` — специальная константа «на CPU».
- Запись с CPU до кернелов (например, `*sum = 0;`) делай **до** prefetch на GPU.
- **CUDA 13:** у `cudaMemPrefetchAsync` другая сигнатура (вместо номера устройства передаётся `cudaMemLocation`, плюс flags). Версию покажет `nvcc --version`. Если там 13.x, используй обёртку `prefetch()` из `midterms/2026-midterm-1-vectors/lastname.cu` или вариант B.

### Вариант B: явные копирования

Отдельные буферы на CPU и на GPU. Для настоящей асинхронности буфер на CPU должен быть **pinned** (`cudaMallocHost`), иначе `cudaMemcpyAsync` работает как обычный `cudaMemcpy`.

```cpp
int *hA, *dA;
CHECK(cudaMallocHost(&hA, bytes));    // pinned host memory
CHECK(cudaMalloc(&dA, bytes));        // GPU memory
CHECK(cudaMemcpyAsync(dA, hA, bytes, cudaMemcpyHostToDevice, s));
kernel<<<blocks, THREADS, 0, s>>>(dA, N);
CHECK(cudaMemcpyAsync(hA, dA, bytes, cudaMemcpyDeviceToHost, s));
CHECK(cudaStreamSynchronize(s));
CHECK(cudaFree(dA));
CHECK(cudaFreeHost(hA));
```

Указатель из `cudaMalloc` нельзя читать на CPU (`dA[0]` в `main` даст segfault). Наоборот, обычный `malloc` нельзя передавать в кернел.

### Чем освобождать (пункт «clear memory»)

| Выделил через | Освободи через |
| --- | --- |
| `cudaMallocManaged` | `cudaFree` |
| `cudaMalloc` | `cudaFree` |
| `cudaMallocHost` | `cudaFreeHost` |
| `malloc` | `free` |
| `cudaStreamCreate` | `cudaStreamDestroy` |
| `cudaEventCreate` | `cudaEventDestroy` |

Размер считай в `size_t`: `size_t bytes = (size_t)N * N * sizeof(int);` — для больших матриц `int` переполнится.

## 5. Ошибки: макрос CHECK

Каждая функция `cuda...` возвращает `cudaError_t`, и без проверки ошибка молча пропадает — а ты получаешь нули. Оборачивай все вызовы в `CHECK`:

```cpp
#define CHECK(call)                                                       \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            printf("CUDA error: %s (%s:%d)\n", cudaGetErrorString(err),   \
                   __FILE__, __LINE__);                                   \
            exit(1);                                                      \
        }                                                                 \
    } while (0)
```

Кернел ничего не возвращает, поэтому после запуска проверяй так:

```cpp
kernel<<<blocks, THREADS>>>(a, N);
CHECK(cudaGetLastError());        // launch errors: bad config, too many threads
CHECK(cudaDeviceSynchronize());   // runtime errors: out-of-bounds access
```

| Сообщение | Причина |
| --- | --- |
| `invalid configuration argument` | threads > 1024 или blocks = 0 |
| `an illegal memory access was encountered` | индекс за границей массива или в кернел передан указатель из `malloc` |
| `out of memory` | N слишком большой |
| `invalid device ordinal` | неправильный номер устройства в prefetch |

Если программа падает или выдаёт мусор, запусти `compute-sanitizer ./serikov` — он покажет кернел и поток, вышедший за границу.

## 6. Streams и синхронизация

**Stream** — это очередь операций на GPU. Внутри одного стрима операции идут строго по порядку, а разные стримы могут работать одновременно. На мидке нужно заполнить массивы «on independent streams»: A в одном стриме, B в другом.

```cpp
cudaStream_t s1, s2;
CHECK(cudaStreamCreate(&s1));
CHECK(cudaStreamCreate(&s2));

fillA<<<blocks, THREADS, 0, s1>>>(a, N);   // 4th parameter = stream
fillB<<<blocks, THREADS, 0, s2>>>(b, N);   // runs concurrently with fillA
CHECK(cudaGetLastError());
CHECK(cudaDeviceSynchronize());            // wait for both before using a and b

CHECK(cudaStreamDestroy(s1));
CHECK(cudaStreamDestroy(s2));
```

Важно из лекции week-5:

- Без указания стрима кернел идёт в **default stream** (стрим 0).
- Default stream блокирующий: он ждёт, пока закончатся все другие стримы, и никто не работает параллельно с ним.
- Работа в разных non-default стримах может перекрываться, порядок между ними не гарантирован.
- `cudaMemPrefetchAsync` и `cudaMemcpyAsync` тоже принимают стрим последним аргументом.

### Какая синхронизация для чего

| Функция | Где вызывать | Чего ждёт |
| --- | --- | --- |
| `cudaDeviceSynchronize()` | host | всей работы на GPU во всех стримах |
| `cudaStreamSynchronize(s)` | host | только стрима `s` |
| `cudaEventSynchronize(e)` | host | пока пройдёт event `e` |
| `__syncthreads()` | внутри кернела | всех потоков **одного блока** (барьер) |

Если сомневаешься — ставь `cudaDeviceSynchronize()`. Лишняя синхронизация стоит миллисекунды, а пропущенная даёт неправильный ответ.

## 7. Замер времени

GPU меряют через **cudaEvent**, CPU — через **std::chrono**. Обычный таймер вокруг запуска кернела покажет почти 0: запуск асинхронный и возвращается сразу.

### GPU: cudaEvent

```cpp
cudaEvent_t start, stop;
CHECK(cudaEventCreate(&start));
CHECK(cudaEventCreate(&stop));

CHECK(cudaEventRecord(start));
mulKernel<<<blocks, THREADS>>>(a, b, c, N);
CHECK(cudaEventRecord(stop));
CHECK(cudaEventSynchronize(stop));          // wait until stop is reached

float gpuMs = 0;
CHECK(cudaEventElapsedTime(&gpuMs, start, stop));
printf("GPU time: %.3f ms\n", gpuMs);

CHECK(cudaEventDestroy(start));
CHECK(cudaEventDestroy(stop));
```

Если кернел запущен в стриме `s`, записывай события в тот же стрим: `cudaEventRecord(start, s)` и `cudaEventRecord(stop, s)`.

### CPU: std::chrono

```cpp
auto t0 = std::chrono::high_resolution_clock::now();
mulHost(a, b, cHost, N);
auto t1 = std::chrono::high_resolution_clock::now();
double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
printf("CPU time: %.3f ms\n", cpuMs);
```

В задании обычно нужно измерить одну операцию (умножение) на большом N, например `2 << 20` (= 2 097 152). Печатай оба времени.

## 8. Shared memory и редукция (week-6)

Редукция — это сумма (или max/min) всего массива на GPU. Она отдельно стоит 3% и есть в каждой задаче, так что её нужно уметь написать по памяти.

### Shared memory и \_\_syncthreads

- `__shared__` — быстрая память **одного блока**: видна всем его потокам, не видна другим блокам, живёт, пока живёт блок.
- Размер статического массива должен быть известен при компиляции: `__shared__ long long cache[THREADS];`, где `THREADS` — `const int`.
- `__syncthreads()` — барьер: ни один поток блока не пойдёт дальше, пока все не дойдут до него. Нужен между «кто-то пишет в shared» и «кто-то другой читает».
- `__syncthreads()` нельзя ставить внутрь `if`, куда заходит только часть потоков: программа зависнет или сломается.

### Алгоритм в 4 шага

1. Каждый поток суммирует свои элементы grid-stride циклом в локальную переменную.
2. Кладёт её в `cache[threadIdx.x]`, потом `__syncthreads()`.
3. Дерево: на каждом шаге первая половина потоков прибавляет себе значение из второй половины, а шаг `s` делится пополам. После каждого шага `__syncthreads()`.
4. В `cache[0]` оказывается сумма блока. Поток 0 добавляет её в общий результат через `atomicAdd`.

Пример дерева для блока из 8 потоков:

| Шаг | cache\[0..7\] |
| --- | --- |
| старт | 3 1 7 0 4 1 6 3 |
| s = 4 (потоки 0–3: `cache[i] += cache[i+4]`) | 7 2 13 3 · · · · |
| s = 2 (потоки 0–1) | 20 5 · · · · · · |
| s = 1 (поток 0) | 25 · · · · · · · |

### Код (заучи наизусть)

```cpp
__global__ void reduceSum(const int *in, long long *result, int n)
{
    __shared__ long long cache[THREADS];
    int tid = threadIdx.x;

    long long sum = 0;                                       // step 1
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        sum += in[i];
    cache[tid] = sum;                                        // step 2
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {           // step 3
        if (tid < s)
            cache[tid] += cache[tid + s];
        __syncthreads();                                     // outside the if!
    }

    if (tid == 0)                                            // step 4
        atomicAdd((unsigned long long *)result, (unsigned long long)cache[0]);
}
```

Вызов:

```cpp
long long *sum;
CHECK(cudaMallocManaged(&sum, sizeof(long long)));
*sum = 0;                                   // must be zero before the kernel
reduceSum<<<blocks, THREADS>>>(c, sum, N);
CHECK(cudaDeviceSynchronize());
printf("Sum = %lld\n", *sum);
```

Правила:

- `blockDim.x` — степень двойки (256), иначе дерево потеряет элементы.
- Сумму держи в `long long`: сумма миллионов произведений переполнит `int`.
- `atomicAdd` для 64-бит есть только для `unsigned long long`, отсюда приведение типа. Отрицательные числа складываются правильно и так (дополнительный код).
- Без атомика тоже можно: каждый блок пишет `partial[blockIdx.x] = cache[0];`, а массив `partial` из `blocks` чисел суммируется на CPU или вторым запуском того же кернела с одним блоком.

### Версии из лекции «Optimizing Parallel Reduction in CUDA»

Все 7 версий складывают одно и то же, отличаются скоростью. Код выше — это версии 3 + 4 + 7. Если спросят, что меняется:

| # | Идея | Проблема, которую решает |
| --- | --- | --- |
| 1 | interleaved addressing: `if (tid % (2*s) == 0)` | — (базовая, с divergent branching) |
| 2 | interleaved со strided index | убирает дивергенцию, но даёт bank conflicts |
| 3 | sequential addressing: `s = blockDim.x/2; s >>= 1` | убирает bank conflicts |
| 4 | first add during load | половина потоков простаивала с первого шага |
| 5 | unroll last warp | последние 32 потока без `__syncthreads` |
| 6 | complete unrolling (templates) | убирает накладные расходы цикла |
| 7 | multiple elements per thread | grid-stride накопление, меньше блоков |

### Max / Min

То же дерево, но `max()` вместо `+`, стартовое значение `INT_MIN` (для min — `INT_MAX`), итог блока через `atomicMax(result, cache[0])`. Результат перед кернелом инициализируй в `INT_MIN`, а не в 0. Полный код — в шаблонах.

### Atomics

Если много потоков пишут в одну ячейку (`*count += 1`), обновления теряются — это **race condition**. Атомарные функции делают «прочитать → изменить → записать» неделимо:

| Функция | Типы | Для чего |
| --- | --- | --- |
| `atomicAdd(&x, v)` | `int`, `unsigned`, `unsigned long long`, `float`, `double` (double — на GPU sm\_60+) | сумма, счётчик, гистограмма |
| `atomicMax(&x, v)` / `atomicMin(&x, v)` | `int`, `unsigned`, 64-битные целые | максимум / минимум |
| `atomicSub`, `atomicExch`, `atomicCAS` | целые | редко нужны |

Атомик на каждый элемент медленный: все потоки стоят в очереди к одной ячейке. Поэтому сначала считают в shared memory внутри блока, а атомик делают один на блок — так устроена и редукция, и гистограмма в шаблонах.

## 9. Матрицы

Матрица N×N хранится одним массивом из N·N элементов строка за строкой (**row-major**). Элемент (строка r, столбец c) лежит по индексу `r * N + c`. Обратно: `r = i / N`, `c = i % N`.

| Что нужно | Индекс |
| --- | --- |
| A\[r\]\[c\] | `A[r * N + c]` |
| транспонированная Bᵀ\[r\]\[c\] = B\[c\]\[r\] | `B[c * N + r]` |
| диагональ (trace) | `A[k * N + k]` для k = 0…N−1 |
| последний элемент C\[N−1\]\[N−1\] | `C[N * N - 1]` |

### Умножение C = A × B

C\[r\]\[c\] — это скалярное произведение строки r матрицы A на столбец c матрицы B: сумма по k от A\[r\]\[k\] × B\[k\]\[c\]. Один поток считает один элемент C, grid-stride идёт по всем N·N элементам:

```cpp
__device__ int dotRowCol(const int *A, const int *B, int n, int row, int col)
{
    int sum = 0;
    for (int k = 0; k < n; k++)
        sum += A[row * n + k] * B[k * n + col];
    return sum;
}

__global__ void matMul(const int *A, const int *B, int *C, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * n; i += blockDim.x * gridDim.x)
        C[i] = dotRowCol(A, B, n, i / n, i % n);
}
```

CPU-версия — три вложенных цикла r, c, k с той же формулой. Она работает за N³: при N = 1024 это уже секунды, при N = 4096 — минуты. Проверяй на умеренном N.

### Вариант с 2D-сеткой

Не обязателен, но полезно узнать его в чужом коде:

```cpp
__global__ void matMul2D(const int *A, const int *B, int *C, int n)
{
    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < n; row += blockDim.y * gridDim.y)
        for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < n; col += blockDim.x * gridDim.x)
            C[row * n + col] = dotRowCol(A, B, n, row, col);
}

dim3 threads2(16, 16);          // 256 threads per block
dim3 blocks2(8, 8);
matMul2D<<<blocks2, threads2>>>(A, B, C, n);
```

Условие «blocks = warp × SM» проще выполнить с 1D-сеткой, поэтому на экзамене бери первый вариант.

### Сумма всех элементов C

Тот же `reduceSum` из раздела 8, только с длиной `N * N`: `reduceSum<<<blocks, THREADS>>>(C, sum, N * N);`

### «Largest array size» (мидка 2)

Три матрицы `int` занимают 3 · N² · 4 байта. Свободную память GPU даёт `cudaMemGetInfo(&freeBytes, &totalBytes)`. Наибольшее N — то, при котором 3 · N² · 4 ещё меньше `freeBytes`; код — в шаблонах. При N > 46 340 произведение `N * N` не влезает в `int`, считай в `size_t`.

## 10. Типичные ошибки

Все здешние ошибки компилятор пропускает — ловит их только `verify`. Если пришло FAILED, иди по таблице сверху вниз.

| Симптом | Причина | Исправление |
| --- | --- | --- |
| все результаты 0 или мусор | нет синхронизации перед чтением на CPU | `cudaDeviceSynchronize()` |
| сумма в blocks раз больше | `stride = blockDim.x` без `gridDim.x` | `blockDim.x * gridDim.x` |
| сумма растёт при каждом запуске редукции | результат не обнулён | `*sum = 0;` перед каждым запуском |
| сумма случайно меняется от запуска к запуску | `+=` в общую ячейку без атомика, или нет `__syncthreads` в дереве | `atomicAdd`, `__syncthreads()` после каждого шага |
| сумма отрицательная огромная | переполнение `int` | `long long`, печать `%lld` |
| max всегда 0 | результат инициализирован 0, а все числа отрицательные | старт с `INT_MIN` |
| редукция теряет часть чисел | threads не степень двойки, или `cache[THREADS]` меньше блока | threads = 256 везде через одну константу |
| зависает | `__syncthreads()` внутри `if` | вынести за `if` |
| `illegal memory access` | выход за границу, `malloc`-указатель в кернеле | условие `i < n`; `cudaMallocManaged`; `compute-sanitizer` |
| segfault на CPU | чтение `cudaMalloc`-указателя в `main` | скопировать назад или использовать managed |
| verify падает после in-place операции | CPU считает по уже изменённым данным | скопируй исходный массив до кернела |
| GPU-время 0.00 ms | таймер chrono вокруг асинхронного запуска | `cudaEvent` + `cudaEventSynchronize` |
| `<<<` — ошибка синтаксиса | файл `.cpp`, а не `.cu` | переименуй в `.cu` |
| вывод после `prefetch` на CPU старый | prefetch асинхронный | `cudaDeviceSynchronize()` после него |

## 11. Шпаргалка API

Все функции, которые нужны для мидки, в одной таблице.

| Задача | Вызов |
| --- | --- |
| номер GPU | `cudaGetDevice(&device)` |
| свойства GPU | `cudaGetDeviceProperties(&prop, device)` → `prop.name`, `prop.multiProcessorCount`, `prop.warpSize` |
| число SM | `cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, device)` |
| размер warp | `cudaDeviceGetAttribute(&w, cudaDevAttrWarpSize, device)` |
| свободная память | `cudaMemGetInfo(&freeB, &totalB)` |
| unified memory | `cudaMallocManaged(&p, bytes)` |
| память GPU | `cudaMalloc(&p, bytes)` |
| pinned память CPU | `cudaMallocHost(&p, bytes)` |
| копирование | `cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice)` / `cudaMemcpyDeviceToHost` |
| асинхронное копирование | `cudaMemcpyAsync(dst, src, bytes, kind, stream)` |
| перенос managed на GPU / CPU | `cudaMemPrefetchAsync(p, bytes, device, stream)` / `(p, bytes, cudaCpuDeviceId)` |
| освобождение | `cudaFree(p)`, `cudaFreeHost(p)`, `free(p)` |
| запуск кернела | `kernel<<<blocks, threads, 0, stream>>>(args)` |
| ошибка запуска | `cudaGetLastError()` |
| текст ошибки | `cudaGetErrorString(err)` |
| ждать всё | `cudaDeviceSynchronize()` |
| стрим | `cudaStreamCreate(&s)`, `cudaStreamSynchronize(s)`, `cudaStreamDestroy(s)` |
| событие | `cudaEventCreate(&e)`, `cudaEventRecord(e, s)`, `cudaEventSynchronize(e)`, `cudaEventElapsedTime(&ms, start, stop)`, `cudaEventDestroy(e)` |
| внутри кернела | `threadIdx.x`, `blockIdx.x`, `blockDim.x`, `gridDim.x`, `warpSize` |
| shared / барьер | `__shared__ T arr[SIZE];`, `__syncthreads()` |
| атомики | `atomicAdd`, `atomicMax`, `atomicMin` |
| printf-форматы | `%d` int, `%lld` long long, `%zu` size\_t, `%.3f` float/double, `%s` строка |

## 12. Шаблоны кода

Шаблон A — полный рабочий файл, который закрывает все пункты грейдинга. Остальные — куски, которые вставляются в него под конкретную задачу. Весь код из этого раздела скомпилирован без ошибок (запускать на GPU я его не мог, прогони на лабе).

### A. Полный каркас: c = a + b и сумма

Формулы заполнения совпадают с задачей 1 тренажёра: правильный ответ `Sum (addition) = -6`.

```cpp
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK(call)                                                       \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            printf("CUDA error: %s (%s:%d)\n", cudaGetErrorString(err),   \
                   __FILE__, __LINE__);                                   \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

const int THREADS = 256;

// ---------- device functions ----------
__device__ int opDevice(int x, int y) { return x + y; }

// ---------- kernels ----------
__global__ void fillA(int *a, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        a[i] = (i % 11) - 5;
}

__global__ void fillB(int *b, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        b[i] = ((i * 7 + 3) % 11) - 5;
}

__global__ void opKernel(const int *a, const int *b, int *c, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        c[i] = opDevice(a[i], b[i]);
}

__global__ void reduceSum(const int *in, long long *result, int n)
{
    __shared__ long long cache[THREADS];
    int tid = threadIdx.x;

    long long sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        sum += in[i];
    cache[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] += cache[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd((unsigned long long *)result, (unsigned long long)cache[0]);
}

// ---------- host functions ----------
void opHost(const int *a, const int *b, int *c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

long long sumHost(const int *c, int n)
{
    long long s = 0;
    for (int i = 0; i < n; i++)
        s += c[i];
    return s;
}

bool verify(const int *gpu, const int *cpu, int n)
{
    for (int i = 0; i < n; i++)
        if (gpu[i] != cpu[i]) {
            printf("Mismatch at %d: GPU %d, CPU %d\n", i, gpu[i], cpu[i]);
            return false;
        }
    return true;
}

void printArray(const char *name, const int *x, int n)
{
    printf("%s = ", name);
    for (int i = 0; i < n; i++)
        printf("%d ", x[i]);
    printf("\n");
}

// ---------- main ----------
int main()
{
    const int N = 2 << 20;
    size_t bytes = N * sizeof(int);

    // 1. device info -> blocks
    int device;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    int numSMs = prop.multiProcessorCount;
    int blocks = prop.warpSize * numSMs;
    printf("Hardware name: %s\n", prop.name);
    printf("SMs = %d, blocks = %d, threads = %d\n", numSMs, blocks, THREADS);

    // 2. memory
    int *a, *b, *c, *cHost;
    long long *sum;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&b, bytes));
    CHECK(cudaMallocManaged(&c, bytes));
    CHECK(cudaMallocManaged(&sum, sizeof(long long)));
    cHost = (int *)malloc(bytes);
    *sum = 0;

    // 3. streams
    cudaStream_t s1, s2;
    CHECK(cudaStreamCreate(&s1));
    CHECK(cudaStreamCreate(&s2));

    // 4. async migration to GPU
    CHECK(cudaMemPrefetchAsync(a, bytes, device, s1));
    CHECK(cudaMemPrefetchAsync(b, bytes, device, s2));
    CHECK(cudaMemPrefetchAsync(c, bytes, device));
    CHECK(cudaMemPrefetchAsync(sum, sizeof(long long), device));

    // 5. fill on independent streams
    fillA<<<blocks, THREADS, 0, s1>>>(a, N);
    fillB<<<blocks, THREADS, 0, s2>>>(b, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // 6. compute + reduce, timed with events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    opKernel<<<blocks, THREADS>>>(a, b, c, N);
    CHECK(cudaEventRecord(stop));
    reduceSum<<<blocks, THREADS>>>(c, sum, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    // 7. async migration back to CPU
    CHECK(cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(sum, sizeof(long long), cudaCpuDeviceId));
    CHECK(cudaDeviceSynchronize());

    // 8. CPU version, timed with chrono
    auto t0 = std::chrono::high_resolution_clock::now();
    opHost(a, b, cHost, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    long long cpuSum = sumHost(cHost, N);

    // 9. output
    printArray("A (first 10)", a, 10);
    printArray("B (first 10)", b, 10);
    printf("CPU results:\nSum (addition) = %lld\n", cpuSum);
    printf("GPU results:\nSum (addition) = %lld\n", *sum);
    printf("Time CPU = %.3f ms, GPU = %.3f ms\n", cpuMs, gpuMs);
    bool ok = verify(c, cHost, N) && (*sum == cpuSum);
    printf("Verify: %s\n", ok ? "PASSED" : "FAILED");

    // 10. clear memory
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaStreamDestroy(s2));
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(c));
    CHECK(cudaFree(sum));
    free(cHost);
    return 0;
}
```

Что менять под другую задачу:

- формулы в `fillA` / `fillB`;
- `opDevice` и `opHost` (для трёх операций — три пары функций и три массива c; сумму обнуляй перед каждой редукцией);
- для матриц: длина `N * N` везде, `opKernel` → `matMul` (шаблон B);
- вывод — под формат из задания (шаблон G).

### B. Умножение матриц (формат мидки 2025)

```cpp
__device__ int dotRowCol(const int *A, const int *B, int n, int row, int col)
{
    int sum = 0;
    for (int k = 0; k < n; k++)
        sum += A[row * n + k] * B[k * n + col];
    return sum;
}

__global__ void matMul(const int *A, const int *B, int *C, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * n; i += blockDim.x * gridDim.x)
        C[i] = dotRowCol(A, B, n, i / n, i % n);
}

void matMulHost(const int *A, const int *B, int *C, int n)
{
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
            int sum = 0;
            for (int k = 0; k < n; k++)
                sum += A[r * n + k] * B[k * n + c];
            C[r * n + c] = sum;
        }
}

// in main: bytes = (size_t)N * N * sizeof(int); every length is N * N
matMul<<<blocks, THREADS>>>(A, B, C, N);
reduceSum<<<blocks, THREADS>>>(C, sum, N * N);
```

### C. Редукция max и min

Нужен `#include <climits>`. Перед запуском: `*mx = INT_MIN; *mn = INT_MAX;`.

```cpp
__global__ void reduceMax(const int *in, int *result, int n)
{
    __shared__ int cache[THREADS];
    int tid = threadIdx.x;
    int m = INT_MIN;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        m = max(m, in[i]);
    cache[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] = max(cache[tid], cache[tid + s]);
        __syncthreads();
    }
    if (tid == 0)
        atomicMax(result, cache[0]);
}

__global__ void reduceMin(const int *in, int *result, int n)
{
    __shared__ int cache[THREADS];
    int tid = threadIdx.x;
    int m = INT_MAX;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        m = min(m, in[i]);
    cache[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] = min(cache[tid], cache[tid + s]);
        __syncthreads();
    }
    if (tid == 0)
        atomicMin(result, cache[0]);
}
```

### D. Редукция без атомиков (два прохода)

Если на лекции показывали именно такой вариант:

```cpp
__global__ void reducePartial(const int *in, long long *partial, int n)
{
    __shared__ long long cache[THREADS];
    int tid = threadIdx.x;
    long long sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        sum += in[i];
    cache[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] += cache[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        partial[blockIdx.x] = cache[0];
}

// host: partial has `blocks` elements (cudaMallocManaged)
reducePartial<<<blocks, THREADS>>>(c, partial, N);
CHECK(cudaDeviceSynchronize());
long long total = 0;
for (int k = 0; k < blocks; k++)
    total += partial[k];
```

### E. Гистограмма и счётчик

```cpp
const int BINS = 11;                       // values -5..5

__global__ void histogram(const int *a, int *hist, int n)
{
    __shared__ int local[BINS];
    for (int j = threadIdx.x; j < BINS; j += blockDim.x)
        local[j] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&local[a[i] + 5], 1);    // shared atomic: fast
    __syncthreads();
    for (int j = threadIdx.x; j < BINS; j += blockDim.x)
        atomicAdd(&hist[j], local[j]);     // one global atomic per bin per block
}

__global__ void countPositive(const int *y, int *count, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        if (y[i] > 0)
            atomicAdd(count, 1);
}
```

### F. Вариант с cudaMalloc + cudaMemcpyAsync

Если managed-память не работает или просят явные копирования:

```cpp
int *hA, *hB, *hC;                       // pinned host buffers
CHECK(cudaMallocHost(&hA, bytes));
CHECK(cudaMallocHost(&hB, bytes));
CHECK(cudaMallocHost(&hC, bytes));
int *dA, *dB, *dC;                       // device buffers
CHECK(cudaMalloc(&dA, bytes));
CHECK(cudaMalloc(&dB, bytes));
CHECK(cudaMalloc(&dC, bytes));

cudaStream_t s;
CHECK(cudaStreamCreate(&s));
CHECK(cudaMemcpyAsync(dA, hA, bytes, cudaMemcpyHostToDevice, s));
CHECK(cudaMemcpyAsync(dB, hB, bytes, cudaMemcpyHostToDevice, s));
addKernel<<<blocks, THREADS, 0, s>>>(dA, dB, dC, N);
CHECK(cudaMemcpyAsync(hC, dC, bytes, cudaMemcpyDeviceToHost, s));
CHECK(cudaStreamSynchronize(s));         // now hC is ready on the CPU

CHECK(cudaStreamDestroy(s));
CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
CHECK(cudaFreeHost(hA)); CHECK(cudaFreeHost(hB)); CHECK(cudaFreeHost(hC));
```

Если массивы заполняются кернелом на GPU, копировать на GPU не нужно — только результаты обратно на CPU.

### G. Вывод в формате мидки 2026

В задании нужно печатать массивы при N = 10 и время при N = 2 << 20. Удобнее всего вынести тело `main` в функцию `void run(int n, bool printArrays)` и вызвать дважды (так сделано в `midterms/2026-midterm-1-vectors/lastname.cu`):

```cpp
run(10, true);         // prints A, B, C arrays and sums
run(2 << 20, false);   // prints sums and CPU / GPU time
```

Строки, которые должны быть в выводе: `Generated arrays when N = 10:`, `A =`, `B =`, `CPU results:`, `C (addition) =` (и subtraction, multiplication), `Sum (addition) =` (и остальные), `Time to complete when N = 2<<20:`, `GPU results:`, `Hardware name:`. В задании сказано, что данные можно генерировать формулой от индекса — это честный способ, `curand` не нужен.

### H. Наибольшее N, которое помещается

```cpp
size_t freeB, totalB;
CHECK(cudaMemGetInfo(&freeB, &totalB));
size_t n = 1;
while (3 * (n * 2) * (n * 2) * sizeof(int) < freeB)
    n *= 2;
printf("Largest array size: N = %zu\n", n);
```

## 13. Чек-лист перед сдачей

Пройдись по списку перед тем, как сдать. Баллы — из грейдинга мидки 2026, штрафы (−10%, −5%, опоздание, лишний код) — из листа мидки 2: в листе мидки 1 их не было, но вряд ли правила другие.

- [ ] Компилируется и запускается (иначе −10%)
- [ ] Файл назван `фамилия.cu` (иначе −5%)
- [ ] В каждом кернеле grid-stride loop с `blockDim.x * gridDim.x` (1%)
- [ ] `blocks = warpSize × число SM` из `cudaGetDeviceProperties` или `cudaDeviceGetAttribute`, `const int THREADS = 256` (1%)
- [ ] A и B заполняются кернелом в двух разных стримах `<<<b, t, 0, s1>>>` / `s2` (1%)
- [ ] `cudaMemPrefetchAsync` на GPU перед кернелами и на CPU перед чтением (1%)
- [ ] На каждый `cudaMalloc*` есть `cudaFree`, стримы и события уничтожены (1%)
- [ ] Отдельная `__device__` функция на каждую операцию (1%)
- [ ] Редукция с `__shared__` и `__syncthreads()` реализована И вызвана для каждой суммы (3% + 1%)
- [ ] Host-версия всех операций и функция `verify`, вывод PASSED
- [ ] Время CPU и GPU для умножения при N = 2 << 20
- [ ] Вывод в формате задания, включая `Hardware name`
- [ ] Нет неиспользуемого кода (на мидке 2 за него снимали 10%): удали ненужные куски шаблона
- [ ] Сдал вовремя (опоздание — минус 1% за каждую минуту)

### Команды терминала

| Что сделать | Команда |
| --- | --- |
| версия CUDA | `nvcc --version` |
| какая GPU и сколько памяти | `nvidia-smi` |
| компиляция | `nvcc -O2 -o serikov serikov.cu` |
| запуск | `./serikov` |
| вывод в файл | `./serikov > out.txt` |
| найти выход за границу и гонки | `compute-sanitizer ./serikov` |
| профиль (посмотреть, идут ли стримы параллельно) | `nsys profile --stats=true ./serikov` |

### Vim по минимуму

| Клавиши | Действие |
| --- | --- |
| `i` / `Esc` | режим вставки / обычный режим |
| `:w` / `:wq` / `:q!` | сохранить / сохранить и выйти / выйти без сохранения |
| `u` / `Ctrl+r` | отменить / вернуть |
| `dd` / `yy` / `p` | удалить строку / скопировать строку / вставить |
| `o` | новая строка ниже и сразу вставка |
| `/text` затем `n` | поиск и следующее совпадение |
| `:42` | перейти на строку 42 (номер из ошибки nvcc) |
| `:set nu` | номера строк |
| `:set paste` | перед вставкой из буфера, чтобы не ломались отступы |
| `gg=G` | выровнять отступы во всём файле |

### План на вечер

1. Прочитай разделы 0–8 (примерно 40 минут).
2. Напечатай шаблон A руками в vim, скомпилируй и запусти на лабе. Правильный ответ — `Sum (addition) = -6`.
3. Реши задачи 1 и 2 в тренажёре, подглядывая в шаблоны.
4. Задачу 4 или 7 — без подсказок, на время, через `exam`.
5. Утром напиши редукцию и `main` из шаблона A по памяти один раз.
