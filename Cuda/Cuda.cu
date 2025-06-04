#include "Cuda.cuh"

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "\
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)



__device__ CudaMatrix d_cudaMatrix;
CudaMatrix* d_cudaMatrix_ptr = nullptr;

__device__ Params d_params;

__device__ CudaScoreGenome* d_scoreGenomes;
CudaScoreGenome* d_scoreGenomes_ptr = nullptr;



float Score(Matrix& matrix, Genome& genome)
{
    float sum = 0;
    for (int i = 0; i < genome.size; i++)
    {
        int from = genome.g[i];
        int to = genome.g[(i + 1) % genome.size];
        sum += matrix.m[from][to];
    }
    return sum;
}

void SimpleSample(Matrix & matrix, Genome & genome)
{
    int n = matrix.size;
    int Start = RandomNumber(0, n);
    vector<bool> Visited(n, false);
    Visited[Start] = true;
    int VisitedCount = 1;

    int i = Start;
    while (VisitedCount <= n)
    {
        genome.g[VisitedCount - 1] = i;
        Visited[i] = true;

        float min1Value = numeric_limits<int>::max();
        int min1Index = -1;
        float min2Value = numeric_limits<int>::max();
        int min2Index = -1;

        for (int j = 0; j < n; j++)
        {
            if (matrix.m[i][j] > 0 && !Visited[j] && matrix.m[i][j] <= min1Value)
            {
                min2Value = min1Value;
                min2Index = min1Index;
                min1Value = matrix.m[i][j];
                min1Index = j;
            }
        }
        int Randomize = RandomNumber(0, 2);
        if (Randomize == 0 && min2Index != -1)
            i = min2Index;
        else
            i = min1Index;
        VisitedCount++;
    }
}

ScoreGenome* FullSimpleSample(Matrix& matrix, Params& params)
{
    ScoreGenome* scoreGenomes = new ScoreGenome[params.maxPopulationWithPadding];

    for (int i = 0; i < params.maxPopulation; i++)
    {
        scoreGenomes[i].genome.g = new int[params.n];
        scoreGenomes[i].genome.size = params.n;
        SimpleSample(matrix, scoreGenomes[i].genome);
        scoreGenomes[i].score = Score(matrix, scoreGenomes[i].genome);
    }

    for (int i = params.maxPopulation; i < params.maxPopulationWithPadding; i++)
    {
        scoreGenomes[i].genome.g = new int[params.n];
        memset(scoreGenomes[i].genome.g, 0, params.n);
        scoreGenomes[i].genome.size = params.n;
        scoreGenomes[i].score = numeric_limits<float>::max();
    }

    std::sort(scoreGenomes, scoreGenomes + params.maxPopulation, [](const ScoreGenome& a, const ScoreGenome& b) {
        return a.score < b.score;
        });

    return scoreGenomes;
}

__global__ void GenerateRandomIntsKernel(int* d_array, int size, int lowerLimit, int upperLimit, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;


    curandState state;
    curand_init(seed, idx, 0, &state);

    int range = upperLimit - lowerLimit;
    int randVal = curand(&state) % range + lowerLimit;
    d_array[idx] = randVal;
}

void GenerateRandomIntsOnGPU(int* d_array, int size, int lowerLimit, int upperLimit, int seedOffset) {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    unsigned long seed = (unsigned long)time(NULL) + seedOffset;;

    GenerateRandomIntsKernel << <blocks, threadsPerBlock >> > (d_array, size, lowerLimit, upperLimit, seed);

    cudaDeviceSynchronize();
}

__device__ float CudaScore(int* d_genome)
{
    float sum = 0;
    for (int i = 0; i < d_params.n; i++)
    {
        int from = d_genome[i];
        int to = d_genome[(i + 1) % d_params.n];
        sum += d_cudaMatrix.m[from * d_params.n + to];
    }
    return sum;
}

__device__ void swap(CudaScoreGenome& a, CudaScoreGenome& b) {
    CudaScoreGenome temp = a;
    a = b;
    b = temp;
}

__global__ void bitonicSortKernel() {
    unsigned int tid = threadIdx.x;

    // Bitonic sort wymaga rozmiaru bêd¹cego potêg¹ 2
    for (int k = 2; k <= d_params.maxPopulationWithPadding; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            unsigned int ixj = tid ^ j;

            if (ixj > tid && ixj < d_params.maxPopulation && tid < d_params.maxPopulation) {
                bool compare = (tid & k) == 0;

                ScoreCompare comp;

                if ((comp(d_scoreGenomes[tid], d_scoreGenomes[ixj]) != compare)) {
                    swap(d_scoreGenomes[tid], d_scoreGenomes[ixj]);
                }
            }

            __syncthreads();  // synchronizacja w¹tku w ka¿dej iteracji
        }
    }
}

__global__ void CrossoverKernel(int* d_mutationRandTable, int iteration)
{
    int c = blockIdx.x;
    int i = threadIdx.x;
    __shared__ int cuttingPoint;
    __shared__ int parent1, parent2, child1, child2;

    if (i == 0)
    {
        int base = iteration * d_params.crossoversPerGeneration * 2;
        parent1 = d_mutationRandTable[base + 2 * c];
        parent2 = d_mutationRandTable[base + 2 * c + 1];
        cuttingPoint = d_mutationRandTable[(parent1 + parent2) % d_params.population];
        if (cuttingPoint == 0) cuttingPoint = 1;
        if (cuttingPoint == d_params.n - 1) cuttingPoint = d_params.n - 2;
        
        //printf("%d %d\n", parent1, parent2);
        child1 = d_params.population + 2 * c;
        child2 = d_params.population + 2 * c + 1;
    }

    __syncthreads();

    __shared__ bool visited1[CROSSOVER_THREADS_PER_BLOCK];
    __shared__ bool visited2[CROSSOVER_THREADS_PER_BLOCK];

    int city1 = d_scoreGenomes[parent1].genome[i];
    int city2 = d_scoreGenomes[parent2].genome[i];
    
    if (i < cuttingPoint)
    {
        d_scoreGenomes[child1].genome[i] = city1;
        d_scoreGenomes[child2].genome[i] = city2;
        visited1[city1] = true;
        visited2[city2] = true;
    }
    else {
        visited1[city1] = false;
        visited2[city2] = false;
    }

    __syncthreads();

    if (i == 0)
    {
        int index = cuttingPoint;
        for (int cityIndex = 0; cityIndex < d_params.n; cityIndex++)
        {
            int city = d_scoreGenomes[parent1].genome[cityIndex];
            if (!visited1[city])
            {
                d_scoreGenomes[child1].genome[index] = city;
                index++;
            }
        }
        d_scoreGenomes[child1].score = CudaScore(d_scoreGenomes[child1].genome);
        //printf("%f\n", d_scoreGenomes[child1].score);
    }

    if (i == 1)
    {
        int index = cuttingPoint;
        for (int cityIndex = 0; cityIndex < d_params.n; cityIndex++)
        {
            int city = d_scoreGenomes[parent2].genome[cityIndex];
            if (!visited2[city])
            {
                d_scoreGenomes[child2].genome[index] = city;
                index++;
            }
        }
        d_scoreGenomes[child2].score = CudaScore(d_scoreGenomes[child2].genome);
    }
    __syncthreads();

}

__global__ void MutationKernel(int* d_mutationRandTable, int iteration)
{
    int m = blockIdx.x;
    int i = threadIdx.x;
    __shared__ int type;
    __shared__ int length, startIndex, endIndex;
    __shared__ int copyStartPoint, copyEndPoint;
    __shared__ int borderLeft, borderRight;

    int randIndex= iteration * d_params.maxPopulation + m;
    int randTableSize = d_params.iterations * d_params.maxPopulation;

    int prop = d_mutationRandTable[randIndex];
    if (i == 0)
    {
        int mutationPropArea = d_params.mutationProp * 10000;
        if (prop > mutationPropArea)
            type = 0;
        else if (prop < mutationPropArea / 2)
            type = 1;
        else
            type = 2;
    }
    __syncthreads();

    if (type == 1 && i == 0)
    {
        int index1 = d_mutationRandTable[(randIndex + prop + 1) % randTableSize] % d_params.n;
        int index2 = d_mutationRandTable[(randIndex + prop + 2) % randTableSize] % d_params.n;

        //printf("%d | %d\n", index1, index2);
        int temp = d_scoreGenomes[m].genome[index1];
        d_scoreGenomes[m].genome[index1] = d_scoreGenomes[m].genome[index2];
        d_scoreGenomes[m].genome[index2] = temp;
        d_scoreGenomes[m].score = CudaScore(d_scoreGenomes[m].genome);
    } 
    else if (type == 2)
    {

        if (i == 0)
        {
            length = d_mutationRandTable[(randIndex + prop + 1) % randTableSize] % (int)(d_params.n / 2);
            if (length < 2) length = 2;
            startIndex = d_mutationRandTable[(randIndex + prop + 1) % randTableSize] % (d_params.n - length);
            copyStartPoint = d_mutationRandTable[(randIndex + prop + 2) % randTableSize] % (d_params.n - length);
            //printf("%d | %d | %d\n", startIndex, copyStartPoint, length);
            endIndex = startIndex + length;
            copyEndPoint = copyStartPoint + length;

            borderLeft = min(copyStartPoint, startIndex);
            borderRight = max(copyEndPoint, endIndex);
      
        }

        __syncthreads();
    
        int value = -1;

        if (i < borderLeft || i >= borderRight)
        {
            value = d_scoreGenomes[m].genome[i];
        }
        else if (i >= copyStartPoint && i < copyEndPoint)
        {
            int offset = i - copyStartPoint;
            value = d_scoreGenomes[m].genome[startIndex + offset];
        }
        else if (i >= copyEndPoint && i <= borderRight)
        {
            value = d_scoreGenomes[m].genome[i - length];
        }
        else if (i >= borderLeft && i < copyStartPoint)
        {
            value = d_scoreGenomes[m].genome[i + length];
        }
  

        __syncthreads();

        d_scoreGenomes[m].genome[i] = value;

        __syncthreads();

        if (i == 0)
        {
            d_scoreGenomes[m].score = CudaScore(d_scoreGenomes[m].genome);
        }
        
    }

    __syncthreads();
    
}

void AllocateCudaMatrix(int size) {
    CudaMatrix h_temp;
    h_temp.size = size;

    size_t totalSize = size * size * sizeof(float);  // Zak³adam macierz kwadratow¹
    CUDA_CHECK(cudaMalloc(&h_temp.m, totalSize));

    // Skopiuj strukturê (z wskaŸnikiem do pamiêci urz¹dzenia) do zmiennej globalnej w GPU
    CUDA_CHECK(cudaMemcpyToSymbol(d_cudaMatrix, &h_temp, sizeof(CudaMatrix)));
}

void FreeCudaMatrix() {
    CudaMatrix h_temp;
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_temp, d_cudaMatrix, sizeof(CudaMatrix)));

    CUDA_CHECK(cudaFree(h_temp.m));
}

void CopyCudaMatrixFromHostToDevice(Matrix& matrix) {
    int n = matrix.size;
    float* host_flat = new float[n * n];

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            host_flat[i * n + j] = matrix.m[i][j];
        }
    }

    CudaMatrix h_temp;
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_temp, d_cudaMatrix, sizeof(CudaMatrix)));

    CUDA_CHECK(cudaMemcpy(h_temp.m, host_flat, n * n * sizeof(float), cudaMemcpyHostToDevice));

    delete[] host_flat;
}

void CopyParamsFromHostToDevice(const Params& h_params) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_params, &h_params, sizeof(Params)));
}

void AllocateCudaScoreGenomes(int maxPopulation, int n) {
    cudaMalloc(&d_scoreGenomes_ptr, maxPopulation * sizeof(CudaScoreGenome));

    CudaScoreGenome* tempArray = new CudaScoreGenome[maxPopulation];

    for (int i = 0; i < maxPopulation; ++i) {
        CUDA_CHECK(cudaMalloc(&(tempArray[i].genome), n * sizeof(int)));
        tempArray[i].score = 0.0f; 
    }

    CUDA_CHECK(cudaMemcpy(d_scoreGenomes_ptr, tempArray, maxPopulation * sizeof(CudaScoreGenome), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(d_scoreGenomes, &d_scoreGenomes_ptr, sizeof(CudaScoreGenome*)));

    delete[] tempArray;
}

void FreeCudaScoreGenomes(int maxPopulation) {
    if (d_scoreGenomes_ptr == nullptr) {
        std::cerr << "Error: Nothing to free (d_scoreGenomes_ptr is null).\n";
        return;
    }

    // Tymczasowo skopiuj tablicê z GPU do hosta, by uzyskaæ dostêp do wskaŸników genome[i]
    CudaScoreGenome* tempArray = new CudaScoreGenome[maxPopulation];
    CUDA_CHECK(cudaMemcpy(tempArray, d_scoreGenomes_ptr, maxPopulation * sizeof(CudaScoreGenome), cudaMemcpyDeviceToHost));

    // Zwolnij ka¿d¹ genome[i]
    for (int i = 0; i < maxPopulation; ++i) {
        CUDA_CHECK(cudaFree(tempArray[i].genome));
    }

    // Zwolnij g³ówn¹ tablicê struktur
    CUDA_CHECK(cudaFree(d_scoreGenomes_ptr));
    d_scoreGenomes_ptr = nullptr;

    delete[] tempArray;
}

void CopyCudaScoreGenomesFromHostToDevice(ScoreGenome* h_scoreGenomes, int maxPopulation) {
    if (d_scoreGenomes_ptr == nullptr) {
        std::cerr << "Error: AllocateCudaScoreGenomes must be called first.\n";
        return;
    }

    CudaScoreGenome* tempArray = new CudaScoreGenome[maxPopulation];
    CUDA_CHECK(cudaMemcpy(tempArray, d_scoreGenomes_ptr, maxPopulation * sizeof(CudaScoreGenome), cudaMemcpyDeviceToHost));

    for (int i = 0; i < maxPopulation; ++i) {
        // Kopiuj genom (tablicê intów)
        CUDA_CHECK(cudaMemcpy(
            tempArray[i].genome,
            h_scoreGenomes[i].genome.g,
            h_scoreGenomes[i].genome.size * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        tempArray[i].score = h_scoreGenomes[i].score;
    }

    CUDA_CHECK(cudaMemcpy(d_scoreGenomes_ptr, tempArray, maxPopulation * sizeof(CudaScoreGenome), cudaMemcpyHostToDevice));

    delete[] tempArray;
}

void AllocateRandTables(Params& params, int** d_crossoversRandTable, int** d_mutationRandTable)
{
    CUDA_CHECK(cudaMalloc(d_crossoversRandTable, sizeof(int) * params.iterations * params.crossoversPerGeneration * 2));    
    CUDA_CHECK(cudaMalloc(d_mutationRandTable, sizeof(int) * params.iterations * params.maxPopulation));

}

void RandTablesInit(Params& params, int* d_crossoversRandTable, int* d_mutationRandTable)
{
    // tablica crossoverow
    GenerateRandomIntsOnGPU(d_crossoversRandTable, params.iterations * params.crossoversPerGeneration * 2, 0, params.n, 200);

    // tablica typow
    GenerateRandomIntsOnGPU(d_mutationRandTable, params.iterations * params.maxPopulation, 0, 10000, 2108);
}

void FreeRandTables(int* d_crossoversRandTable, int* d_mutationRandTable)
{
    CUDA_CHECK(cudaFree(d_crossoversRandTable));
    CUDA_CHECK(cudaFree(d_mutationRandTable));
}

ScoreGenome CopyScoreGenomeFromDeviceToHost(int index, int n) {
    ScoreGenome result;

    CudaScoreGenome tempCudaGenome;
    CUDA_CHECK(cudaMemcpy(&tempCudaGenome, d_scoreGenomes_ptr + index, sizeof(CudaScoreGenome), cudaMemcpyDeviceToHost));

    result.score = tempCudaGenome.score;
    result.genome.size = n;
    result.genome.g = new int[n];

    CUDA_CHECK(cudaMemcpy(result.genome.g, tempCudaGenome.genome, n * sizeof(int), cudaMemcpyDeviceToHost));

    return result;
}


ScoreGenome CudaGenetic(Matrix &matrix, Settings settings)
{
    Params params;
    params.n = matrix.size;
    params.iterations = settings.iterations;
    params.crossoversPerGeneration = settings.crossoversPerGenerations;
    params.population = settings.population;
    params.maxPopulation = settings.population + settings.crossoversPerGenerations * 2;
    params.mutationProp = settings.mutationProp;

    int pow2 = 1;
    while (pow2 < params.maxPopulation) pow2 <<= 1; // dopelnianie do potegi dwojki
    params.maxPopulationWithPadding = pow2;


    ScoreGenome* scoreGenomes = FullSimpleSample(matrix, params);

    cout << "Starting score: " << scoreGenomes[0].score << endl;

    // sekcja alokacji W GPU
    AllocateCudaMatrix(params.n * params.n);
    CopyCudaMatrixFromHostToDevice(matrix);

    CopyParamsFromHostToDevice(params);

    AllocateCudaScoreGenomes(params.maxPopulationWithPadding, params.n);
    CopyCudaScoreGenomesFromHostToDevice(scoreGenomes, params.maxPopulationWithPadding);

    int * d_crossoversRandTable, * d_mutationRandTable, * d_randT1, * d_randT2, * d_randT3;
    AllocateRandTables(params, &d_crossoversRandTable, &d_mutationRandTable);
    RandTablesInit(params, d_crossoversRandTable, d_mutationRandTable);
    // W³aœciwy algorytm

    for (int generation = 0; generation < settings.iterations; generation++)
    {
        //cout << generation << endl;

        CrossoverKernel << < params.crossoversPerGeneration, params.n >> > (d_crossoversRandTable, generation);
        MutationKernel << < params.maxPopulation, params.n >> > (d_mutationRandTable, generation);
        //cin.get();

        bitonicSortKernel << <1, 128 >> > ();
        cudaDeviceSynchronize();

    }


    FreeCudaMatrix();

    ScoreGenome result = CopyScoreGenomeFromDeviceToHost(0, params.n);

    FreeCudaScoreGenomes(params.maxPopulation);


    for (int i = 0; i < params.maxPopulation; i++)
    {
        freeGenome(scoreGenomes[i].genome);
    }
    delete[] scoreGenomes;


    return result;
}