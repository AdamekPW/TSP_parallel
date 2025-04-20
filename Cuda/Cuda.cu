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


int RandomNumber(int lowerLimit, int upperLimit)
{
    std::random_device rd;  // generator losowoœci (zwykle bazuj¹cy na sprzêcie)
    std::mt19937 gen(rd()); // silnik Mersenne Twister
    std::uniform_int_distribution<> distrib(lowerLimit, upperLimit - 1);
    return distrib(gen);
}

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
    ScoreGenome* scoreGenomes = new ScoreGenome[params.maxPopulation];

    for (int i = 0; i < params.maxPopulation; i++)
    {
        scoreGenomes[i].genome.g = new int[params.n];
        scoreGenomes[i].genome.size = params.n;
        SimpleSample(matrix, scoreGenomes[i].genome);
        scoreGenomes[i].score = Score(matrix, scoreGenomes[i].genome);
    }

    std::sort(scoreGenomes, scoreGenomes + params.population, [](const ScoreGenome& a, const ScoreGenome& b) {
        return a.score < b.score;
        });

    return scoreGenomes;
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

__device__ uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ int GetRandomInt(int lower, int upper, uint32_t seed) {
    uint32_t state = seed; 
    uint32_t randVal = xorshift32(state);

    int range = upper - lower;
    int randInt = lower + (randVal % range);
    return randInt;
}

__device__ void swap(CudaScoreGenome& a, CudaScoreGenome& b) {
    CudaScoreGenome temp = a;
    a = b;
    b = temp;
}

__global__ void bitonicSortKernel() {
    unsigned int tid = threadIdx.x;

    // Bitonic sort wymaga rozmiaru bêd¹cego potêg¹ 2
    for (int k = 2; k <= d_params.maxPopulation; k <<= 1) {
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

__global__ void CrossoverKernel()
{
    int c = blockIdx.x;
    int i = threadIdx.x;
    __shared__ int cuttingPoint;
    __shared__ int parent1, parent2, child1, child2;

    if (i == 0)
    {
        cuttingPoint = GetRandomInt(1, d_params.n - 1, clock64()+c*2);
        parent1 = GetRandomInt(1, d_params.population - 1, clock64() + c * 3);
        parent2 = GetRandomInt(1, d_params.population - 1, clock64() + c * 4);
        while (parent1 == parent2) parent2 = GetRandomInt(1, d_params.population - 1, clock64() + c * 1 + 2);
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
        for (int cityIndex = 0; cityIndex < CROSSOVER_THREADS_PER_BLOCK; cityIndex++)
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
        for (int cityIndex = 0; cityIndex < CROSSOVER_THREADS_PER_BLOCK; cityIndex++)
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

//inline __device__ int min(int a, int b)
//{
//    if (a < b) return a;
//    return b;
//}
//
//inline __device__ int max(int a, int b)
//{
//    if (a > b) return a;
//    return b;
//}

__global__ void MutationKernel()
{
    int m = blockIdx.x;
    int i = threadIdx.x;
    int mutationArea = d_params.mutationProp * 10000;
    __shared__ int p;
    if (i == 0)
    {
        p = GetRandomInt(0, 10000, clock64() + m * 2 + 1);
    }
    __syncthreads();
    if (p < mutationArea)
    {
        
        __shared__ int length, startIndex, endIndex;
        __shared__ int copyStartPoint, copyEndPoint;
        __shared__ int borderLeft, borderRight;

        if (i == 0)
        {
            length = GetRandomInt(2, (int)(d_params.n / 2), clock64() + m * 7 + 3);
            startIndex = GetRandomInt(0, d_params.n - length, clock64() + m * 3 + 1);
            endIndex = startIndex + length;
            copyStartPoint = GetRandomInt(0, d_params.n - length, clock64() + m * 5 + 2);
            copyEndPoint = copyStartPoint + length;

            borderLeft = min(copyStartPoint, startIndex);
            borderRight = max(copyEndPoint, endIndex);

            //printf("length: %d\n", length);
            //printf("startIndex: %d\n", startIndex);
            //printf("endIndex: %d\n", endIndex);
            //printf("copyStartPoint: %d\n", copyStartPoint);
            //printf("copyEndPoint: %d\n", copyEndPoint);
            //printf("borderLeft: %d\n", borderLeft);
            //printf("borderRight: %d\n\n", borderRight);

       
            //for (int j = 0; j < d_params.n; j++)
            //{
            //    if (j == startIndex || j == endIndex) printf(" | ");
            //    if (j == borderLeft || j == borderRight) printf(" || ");
            //   
            //    printf("%d ", d_scoreGenomes[m].genome[j]);
            //}
            //printf("\n");
      
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


        /*if (i == 0)
        {
            for (int j = 0; j < d_params.n; j++)
            {
                if (j == borderLeft || j == borderRight) printf(" || ");
                if (j == copyStartPoint|| j == copyEndPoint) printf(" | ");
                printf("%d ", d_scoreGenomes[m].genome[j]);
            }
            printf("\n");
            bool visited[CROSSOVER_THREADS_PER_BLOCK];
            for (int j = 0; j < d_params.n; j++)
            {
                visited[j] = false;
            }
            for (int j = 0; j < d_params.n; j++)
            {
                int city = d_scoreGenomes[m].genome[j];
                visited[city] = true;
            }
            int sum = 0;
            for (int j = 0; j < d_params.n; j++)
            {
                if (!visited[j])
                {
                    printf("Nie znaleziono: %d \n", j);
                    sum++;
                }
            }
            printf("total: %d\n", sum);
        }*/
    }

    __syncthreads();
    /*
    if (p < (int)(mutationArea / 2))
    {
        int index1 = GetRandomInt(0, d_params.n, clock64() + i);
        int index2 = GetRandomInt(0, d_params.n, clock64() + 2*i+1);


        //printf("%d | %d  %d\n", i, index1, index2);
        int temp = d_scoreGenomes[i].genome[index1];
        d_scoreGenomes[i].genome[index1] = d_scoreGenomes[i].genome[index2];
        d_scoreGenomes[i].genome[index2] = temp;
    }
    else {
        int length = GetRandomInt(2, (int)(d_params.n / 2), clock64() + 3 * i + 8);
        int startIndex = GetRandomInt(0, d_params.n - length, clock64() + 2 * i + 1);
        int endIndex = startIndex + length;

        int copyStartPoint = GetRandomInt(0, d_params.n - length, clock64() + 5 * i + 9);
        int copyEndPoint = copyStartPoint + length;

        int newGenome[CROSSOVER_THREADS_PER_BLOCK];
        bool visited[CROSSOVER_THREADS_PER_BLOCK];

        int index = copyStartPoint;
        for (int j = startIndex; j < endIndex; j++)
        {
            int city = d_scoreGenomes[i].genome[j];
            newGenome[index] = city;
            visited[city] = true;
            index++;
        }



    } */
  
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
        for (int j = 0; j < n; ++j)
            host_flat[i * n + j] = matrix.m[i][j];

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
    params.crossoversPerGeneration = settings.crossoversPerGenerations;
    params.population = settings.population;
    params.maxPopulation = settings.population + settings.crossoversPerGenerations * 2;
    params.mutationProp = settings.mutationProp;

    ScoreGenome* scoreGenomes = FullSimpleSample(matrix, params);

    cout << "Starting score: " << scoreGenomes[0].score << endl;

    // sekcja alokacji W GPU
    AllocateCudaMatrix(params.n * params.n);
    CopyCudaMatrixFromHostToDevice(matrix);

    CopyParamsFromHostToDevice(params);

    AllocateCudaScoreGenomes(params.maxPopulation, params.n);
    CopyCudaScoreGenomesFromHostToDevice(scoreGenomes, params.maxPopulation);

    // W³aœciwy algorytm

    for (int generation = 0; generation < settings.iterations; generation++)
    {
        //cout << generation << endl;

        CrossoverKernel << < params.crossoversPerGeneration, params.n >> > ();
        MutationKernel << < params.maxPopulation, params.n>> > ();
        //cin.get();

        bitonicSortKernel << <1, 128 >> > ();
        cudaDeviceSynchronize();

    }


    // sekcja czyszczenia w GPU
    FreeCudaMatrix();

    /*for (int i = 0; i < params.maxPopulation; i++)
    {
        ScoreGenome r = CopyScoreGenomeFromDeviceToHost(i, params.n);
        cout << i << " : " << r.score << endl;
    }*/
    ScoreGenome result = CopyScoreGenomeFromDeviceToHost(0, params.n);

    FreeCudaScoreGenomes(params.maxPopulation);


    for (int i = 0; i < params.maxPopulation; i++)
    {
        freeGenome(scoreGenomes[i].genome);
    }
    delete[] scoreGenomes;

    return result;
}