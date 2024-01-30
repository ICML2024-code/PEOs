#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <time.h>
//#include <cstdlib>
#include "hnswlib/hnswlib.h"
#include <omp.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <boost/math/distributions/normal.hpp>
//#include <chrono>

#include <unordered_set>

/*
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else
#include <x86intrin.h>
#endif
#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
*/
using namespace std;
using namespace hnswlib;
//using namespace cv;

class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }

};

struct k_elem{
	int id;
	float dist;
};

int QsortComp(				// compare function for qsort
	const void* e1,						// 1st element
	const void* e2)						// 2nd element
{
	int ret = 0;
	k_elem* value1 = (k_elem *) e1;
	k_elem* value2 = (k_elem *) e2;
	if (value1->dist < value2->dist) {
		ret = -1;
	} else if (value1->dist > value2->dist) {
		ret = 1;
	} else {
		if (value1->id < value2->id) ret = -1;
		else if (value1->id > value2->id) ret = 1;
	}
	return ret;
}

int compare_int(const void *a, const void *b)
{
    return *(int*)a - *(int*)b;
}

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
static size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
    {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}

float compare2(const float* a, const float* b, unsigned size) {
	float res = 0;
	for (int i = 0; i < size; i++) {
		float x = a[i] - b[i];
		float t = x * x;
		res += t;
	}
	return (res);
}


static void
get_gt(unsigned int *massQA, float *massQ, float *mass, size_t vecsize, size_t qsize, L2Space &l2space,
       size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {


    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
//    cout << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[100 * i + j]);
        }
    }
}

static float
test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg,
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, 
			 float*** lsh_vec, float* thres_pos, size_t vecdim, 
			float* query_rot, float** query_lsh, int* permutation, int table_size) {
    size_t correct = 0;
    size_t total = 0;

//#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < qsize; i++) {
        unsigned int* result = new unsigned int[k];
		float* tmp = massQ + vecdim * i;

        
        for(int j = 0; j < vecdim; j++){
			int x = permutation[j];
			query_rot[j] = tmp[x];
		}
        
		
        appr_alg.searchKnn(query_rot, k, result, lsh_vec, thres_pos, query_lsh, table_size);
		
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size()) {

            g.insert(gt.top().second);
            gt.pop();
        }

        for(int j = 0; j < k; j++) {
            if (g.find(result[j]) != g.end()) {

                correct++;
            } 
        }
    
		delete[] result;
    }
    return 1.0f * correct / total;
}

static void
test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, float*** lsh_vec, float* thres_pos, float* query_rot, float** query_lsh, int* permutation, int table_size) {
    vector<size_t> efs;// = { 10,10,10,10,10 };
/* 
    for (int i = 10; i < 100; i += 10) {
        efs.push_back(i);
    }
*/
    for (int i = 100; i < 5000; i += 50) {
        efs.push_back(i);
    }
	
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        StopW stopw = StopW();

	//	auto e = std::chrono::high_resolution_clock::now(); // newly added
        float recall = test_approx(massQ, vecsize, qsize, appr_alg, answers, k, lsh_vec,  thres_pos, vecdim, query_rot, query_lsh, permutation, table_size);
	//		auto s = std::chrono::high_resolution_clock::now(); //newly added
	//		std::chrono::duration<double> diff = e - s;  //newly added 
		//   std::cout << "search time: " << diff.count() << "\n";  //newly added 
		
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;

        cout << ef << "\t" << recall << "\t" << 1e6 / time_us_per_query << " QPS\n";
        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

float uniform(						// r.v. from Uniform(min, max)
	float min,							// min value
	float max)							// max value
{
	if (min > max) {printf("input error\n"); exit(0);}

	float x = min + (max - min) * (float) rand() / (float) RAND_MAX;
	if (x < min || x > max) {printf("input error\n"); exit(0);}

	return x;
}

float gaussian(						// r.v. from Gaussian(mean, sigma)
	float mu,							// mean (location)
	float sigma)						// stanard deviation (scale > 0)
{
	float PI = 3.141592654F;
    float FLOATZERO = 1e-6F;
	
	if (sigma <= 0.0f) {printf("input error\n"); exit(0);}

	float u1, u2;
	do {
		u1 = uniform(0.0f, 1.0f);
	} while (u1 < FLOATZERO);
	u2 = uniform(0.0f, 1.0f);
	
	float x = mu + sigma * sqrt(-2.0f * log(u1)) * cos(2.0f * PI * u2);
	//float x = mu + sigma * sqrt(-2.0f * log(u1)) * sin(2.0f * PI * u2);
	return x;
}

float calc_quantile(float expectation, float variance, float thres){
    boost::math::normal_distribution<> normal(expectation, variance);
    return boost::math::quantile(normal, 1 - thres);	 //lambda	
}


void PEOs(int efc_, int M_, int data_size_, int query_size_, int dim_, int table_size, char* path_q_, char* path_data_, char* truth_data_, int L_, float eps_, int topk_) {
	//printf("threads = %d\n", omp_get_max_threads());
	int efConstruction = efc_;
	int M = M_;
    int maxk = 100;
    size_t vecsize = data_size_;

    size_t qsize = query_size_;
    size_t vecdim = dim_;
	int step = table_size;
	size_t true_vecdim = dim_;
    char path_index[1024];
    //char path_gt[1024];
    char *path_q = path_q_;
    char *path_data = path_data_;
    sprintf(path_index, "index.bin");

    int m = 128;
	int level = L_;
	int LSH_level = L_;
	int vecdim0 = vecdim / level;
    int LSH_vecdim0 = vecdim / LSH_level;

	float* R = new float [vecdim * vecdim];

    float*** LSH_vec = new float** [LSH_level];
	for(int i = 0; i < LSH_level; i++)
		LSH_vec[i] = new float* [m];

    for(int i = 0; i < LSH_level; i++){
	    for(int j = 0; j < m; j++){		
		    LSH_vec[i][j] = new float[LSH_vecdim0];
	    }	
    }
	
    int train_size = 100000;	
	float init_val = 1.0f / table_size;
	float thres;
	
	int thres_num = 10;
	float** thres_pos = new float*[thres_num];
	for(int i = 0; i < thres_num; i++){
		thres_pos[i] = new float[step];
	}
	
	
	float min_norm0, max_norm0, diff0;
	
	float coeff = sqrt(2 * log(m) * LSH_level);

    float *massb = new float[vecdim];

    cout << "Loading GT:\n";
    ifstream inputGT(truth_data_, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * maxk];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputGT.read((char *) &t, 4);
        inputGT.read((char *) (massQA + 100 * i), 4 * maxk);
    }
    inputGT.close();
	
    cout << "Loading queries:\n";
    float *massQ = new float[qsize * vecdim];
    ifstream inputQ(path_q, ios::binary);

    for (int i = 0; i < qsize; i++) {
        int in = 0;
        inputQ.read((char *) &in, 4);
        inputQ.read((char *) massb, 4 * true_vecdim);

        for (int j = 0; j < vecdim; j++) {
            massQ[i * vecdim + j] = massb[j];
        }
    }
    inputQ.close();

    float *mass = new float[vecdim];
    ifstream input(path_data, ios::binary);
    int in = 0;
    L2Space l2space(vecdim);
	InnerProductSpace ipsubspace(vecdim0);
	InnerProductSpace LSHsubspace(LSH_vecdim0);
	InnerProductSpace ipspace(vecdim);  //new
    int* permutation = new int[vecdim];	

	float read_diff2;	

    HierarchicalNSW<float> *appr_alg;
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
		
		
		ifstream input2("ProjInfo", ios::binary);


		input2.read((char*)(&read_diff2), sizeof(float));	
		
	    for(int j = 0; j < LSH_level; j++){
		for(int i = 0; i < m; i++){
            input2.read((char*) LSH_vec[j][i] , sizeof(float) * LSH_vecdim0);
		}	
		}		
		
		input2.read((char*)(permutation), sizeof(int) * vecdim);
	    
		for(int j = 0; j < thres_num; j++)			
        input2.read((char*) thres_pos[j], sizeof(float) * step); 
		
        appr_alg = new HierarchicalNSW<float>(&l2space, &ipsubspace, &ipspace, &LSHsubspace, path_index, read_diff2, LSH_level, LSH_vecdim0, false);
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    } else {
        cout << "Building HNSW index:\n";	
//--------------load the dataset----------------------
        float** vec = new float*[vecsize];
	    for(int i = 0; i < vecsize; i++)
		    vec[i] = new float[vecdim];

        for (int i = 0; i < vecsize; i++) {        
            input.read((char *) &in, 4);
            input.read((char *) vec[i], 4 * true_vecdim);	
	    }

//----------------compute norms--------------------
        unsigned short int* norm_quan = new unsigned short int[vecsize];			
		float* norm = new float[vecsize];
        float* true_norm = new float[vecsize];		
		
		for(int i = 0; i < vecsize; i++)
			norm[i] = 0;
		
		for(int i = 0; i < vecsize; i++){
		    for(int k = 0; k < level; k++){
			    float sum = 0;
			    for(int j = 0; j < vecdim0; j++){
			        sum += vec[i][k * vecdim0 + j] * vec[i][k * vecdim0 + j];	
			    }
			    norm[i] += sum; 			
		    }

            true_norm[i] = sqrt(norm[i]);
            norm[i] = norm[i] / 2;					

			if(i == 0) {min_norm0 = norm[i]; max_norm0 = norm[i];}
			else{
				if(norm[i] > max_norm0) {max_norm0 = norm[i];}
				if(norm[i] < min_norm0) {min_norm0 = norm[i];}
			}		
		}

	//	printf("min_norm = %f, max = %f\n", min_norm0 * 2, max_norm0 * 2);
		
		int interval = 256*256;
		unsigned short int b;

		diff0 = (max_norm0 - 0) / (interval - 1);
		for(int i = 0; i < vecsize; i++){
			int a = (norm[i] - 0) / diff0;
		    if(a < 0) {b = 0;} 
            else if(a > 65535) {
				b = 65535;
			}
            else{
				b = a;
			}
            norm_quan[i] = b;	 		
		}
						
        appr_alg = new HierarchicalNSW<float>(m, diff0, level, vecdim0, LSH_level, LSH_vecdim0, &l2space, &ipsubspace, &ipspace, &LSHsubspace, vecsize, M, efConstruction);

//---------------compute projected values-------------------------	
    for(int j = 0; j < thres_num; j++){
		thres = 1 - (0.05 * (j + 1));
        for(int i = 0; i < step; i++){   //skip revision
		    float val = i * init_val;
		    thres_pos[j][i] = calc_quantile(val*coeff, 1- (val * val), thres);

			if(i == 0) thres_pos[j][i] = -100000000;
			if(i == step-1) thres_pos[j][i] = 100000000;	
	
	    }	
	}	
		for(int i = 0; i < LSH_level; i++){ 
	        for(int j = 0; j < m; j++){		
		        for(int l = 0; l < LSH_vecdim0; l++){
			        LSH_vec[i][j][l]= gaussian(0.0f, 1.0f);
		        }
	        }
		}
	
//-----------------------------------------------------------------

        appr_alg->addPoint((void *) vec[0], (size_t) 0, &(norm[0]));
	
        int j1 = 0;
        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = 10000;

#pragma omp parallel for schedule(dynamic)
        for (int i = 1; i < vecsize; i++) {
	    //float* mass = new float[vecdim];
            int j2=0;
#pragma omp critical
            {								
                j1++;
                j2=j1;
                if (j1 % report_every == 0) {
                    cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips " << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
            appr_alg->addPoint((void *) vec[j2], (size_t) j2, &(norm[j2]));
        }
		cout << "HNSW build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";

//--------------------rotation------------------------------------

StopW stopw_rotation = StopW();

        double* norm2 = new double[vecdim];
		//double* avg_norm = new double[vecdim];
		
		k_elem* sort_arr = new k_elem[vecdim];
		k_elem* cur_sum = new k_elem[level];
		for(int i = 0; i < level; i++){
			cur_sum[i].id = i;
			cur_sum[i].dist = 0;
		}
		
		k_elem* temp = new k_elem[level];
		int** id_arr = new int*[level];
		for(int i = 0; i < level; i++)
			id_arr[i] = new int[vecdim0];

        size_t edge_count = 0;
		for(int i = 0; i < vecdim; i++) norm2[i] = 0;
		
		size_t* edge_size = new size_t[vecsize];
		for(int i = 0; i < vecsize; i++){
			edge_size[i] = 0;
		}
		double** edge_norm = new double*[vecsize];
		for(int i = 0; i < vecsize; i++){
			edge_norm[i] = new double[vecdim];
			for(int j = 0; j < vecdim; j++){
				edge_norm[i][j] = 0;
			}
		}
		
//#pragma omp parallel for
		
#pragma omp parallel for schedule(dynamic)
		for(int i = 0; i < vecsize; i++){ 
	        appr_alg->CalcEdgeNorm(i, vecdim, edge_norm[i], &(edge_size[i]));
		}	

        for(int i = 0; i < vecsize; i++)		
            edge_count += edge_size[i];

        for(int i = 0; i < vecsize; i++){
            for(int j = 0; j < vecdim; j++){		
                norm2[j] += edge_norm[i][j] / edge_count;	
			}				
	    }
		
		//----------test permutation start-----------------
        for(int i = 0; i < level; i++){
			int ss = i * vecdim0;
			double ssum = 0;
		    for(int j = 0; j < vecdim0; j++){
				ssum += norm2[ ss + j];
		    }
		}		
		//----------test permutation end-------------------

		
		for(int i = 0; i < vecdim; i++){
			sort_arr[i].id = i;
			sort_arr[i].dist = (float)(norm2[i]);
		}

        qsort(sort_arr, vecdim, sizeof(k_elem), QsortComp);

        int half_dim = vecdim0 / 2;
		
		int t = 0;
        for(int i = 0; i < half_dim; i++){
			for(int j = 0; j < level; j++){
				temp[j].dist = sort_arr[i * level + j].dist + sort_arr[vecdim - 1 - (i * level + j)].dist;
			    temp[j].id = j;
			}
			
			qsort(temp, level, sizeof(k_elem), QsortComp);
		    for(int j = 0; j < level; j++){
			    cur_sum[j].dist = cur_sum[j].dist + temp[level - 1 - j].dist;
			    int l = cur_sum[j].id;
				int k = temp[level - 1 - j].id;
			    id_arr[l][t] = sort_arr[i * level + k].id;
				id_arr[l][t+1] =sort_arr[vecdim - 1 - (i * level + k)].id;
			}
			t += 2;
			qsort(cur_sum, level, sizeof(k_elem), QsortComp);
		}
		
		if(vecdim0 % 2 != 0){
			int i = vecdim0 / 2;
			for(int j = 0; j < level; j++){
				temp[j].dist = sort_arr[i * level + j].dist;
			    temp[j].id = j;
			}
			
			qsort(temp, level, sizeof(k_elem), QsortComp);
		    for(int j = 0; j < level; j++){
			    cur_sum[j].dist = cur_sum[j].dist + temp[level - 1 - j].dist;
			    int l = cur_sum[j].id;
				int k = temp[level - 1 - j].id;
			    id_arr[l][t] = sort_arr[i * level + k].id;
			}
			t += 1;			
		}

        for(int i = 0; i < level; i++){
			qsort(id_arr[i], vecdim0, sizeof(int), compare_int);
		}
		
		for(int i = 0; i < level; i++){
			for(int j = 0; j < vecdim0; j++){
				permutation[i * vecdim0 + j] = id_arr[i][j];
			}
		}
			
		float* tmp_arr = new float[vecdim];
		
		for(int i = 0; i < vecsize; i++){			
			float* tmp = vec[i];
		    for(int j = 0; j < vecdim; j++){
			    int x = permutation[j];
			    tmp_arr[j] = tmp[x];
		    }
		    for(int j = 0; j < vecdim; j++){
			    tmp[j] = tmp_arr[j];
		    }						
		}

		for(int i = 0; i < vecdim; i++){
			int count_ = 0;
			for(int j = 0; j < vecdim; j++){
				if(permutation[j] == i){
					count_++;
				}				
			}
			if(count_ != 1){
				printf("error count = %d, count_\n");
				exit(0);
			}
			
		}

//#pragma omp parallel for

#pragma omp parallel for schedule(dynamic)
 	for(int i = 0; i < vecsize; i++)			
		appr_alg->PermuteVec(i, vec, vecdim);

    for(int i = 0; i < vecsize; i++) {delete[] edge_norm[i];}
	
	delete[] edge_norm;
	delete[] edge_size;

    cout << "Rotation time:" << 1e-6 * stopw_rotation.getElapsedTimeMicro() << "  seconds\n";
	
//	appr_alg->Calc_wres(edge_count, vecdim0, vecdim, vecsize, level);

        StopW stopw_full2 = StopW();

		float** tmp_norm2 = new float*[vecsize];
		for(int i = 0; i < vecsize; i++)
			tmp_norm2[i] = new float[2 * M];
		
		float** tmp_adjust = new float*[vecsize];
		for(int i = 0; i < vecsize; i++)
			tmp_adjust[i] = new float[2 * M];

		float** tmp_res = new float*[vecsize];
		for(int i = 0; i < vecsize; i++)
			tmp_res[i] = new float[2 * M];	

		float** tmp_last = new float*[vecsize];
		for(int i = 0; i < vecsize; i++)
			tmp_last[i] = new float[2 * M];

		bool** is_edge = new bool*[vecsize];
		for(int i = 0; i < vecsize; i++)
			is_edge[i] = new bool[2 * M];			
		
		float* max_norm2 = new float[vecsize];
		float* max_adjust = new float[vecsize];
		float* max_res = new float[vecsize];
		float* max_last = new float[vecsize];
		
		bool* is_zero = new bool[vecsize];
		for(int i = 0; i < vecsize; i++) is_zero[i] = false;
		
		j1 = 0;
		
		//-------test------------------
		float* test_val = new float[vecsize];
		for(int i = 0; i < vecsize; i++) test_val[i] = 0;
		//------------------------------
				
//      #pragma omp parallel for
		

#pragma omp parallel for schedule(dynamic)
	    for (int i = 0; i < vecsize; i++){
			int j2 = 0;
#pragma omp critical
            {								
                j1++;
                j2=j1;
            }
			
	        appr_alg->addProjVal(i, LSH_vec, tmp_norm2[i], tmp_adjust[i], tmp_res[i], tmp_last[i], vecdim, &(max_norm2[i]), &(max_adjust[i]), &(max_res[i]), &(max_last[i]), norm_quan, &(test_val[i]), &(is_zero[i]), is_edge[i]);
	    }
	
        //float tol_max_norm = max_norm[0]; 
		float tol_max_norm2 = max_norm2[0];
        float tol_max_adjust = max_adjust[0]; 
		float tol_max_res = max_res[0];	
        float tol_max_last = max_last[0];			
		
		
        for(int i = 1; i < vecsize; i++){
            //if(max_norm[i] > tol_max_norm) {tol_max_norm = max_norm[i];}  
            if(max_norm2[i] > tol_max_norm2) {tol_max_norm2 = max_norm2[i];}
            if(max_adjust[i] > tol_max_adjust) {tol_max_adjust = max_adjust[i];}  
            if(max_res[i] > tol_max_res) {tol_max_res = max_res[i];}
            if(max_last[i] > tol_max_last) {tol_max_last = max_last[i];}			
 			
		}
		
		delete[] max_norm2;

        int interval0 = 32767;
        //float diff = (tol_max_norm - 0) / (interval - 1);
		float diff2 = (tol_max_norm2 - 0) / (interval - 1);
		float diffadj = (tol_max_adjust - 0) / (interval0 - 1);
		float diffres = (tol_max_res - 0) / (interval - 1);
		float difflast = (tol_max_last - 0) / (interval0 - 1);


	    for (int i = 0; i < vecsize; i++){
	        appr_alg->addEdgeNorm(i, tmp_norm2[i], tmp_adjust[i], tmp_res[i], tmp_last[i], diff2, diffadj, diffres, difflast, is_edge[i]);
	    }	

	    appr_alg->compression(vecsize, vecdim, is_zero);
	    		
        input.close();
        cout << "Projection time:" << 1e-6 * stopw_full2.getElapsedTimeMicro() << "  seconds\n";
		
		
		read_diff2 = diff2;
		
        appr_alg->saveIndex(path_index, read_diff2);
		
		ofstream output("ProjInfo", ios::binary);

		output.write((char*)(&read_diff2), sizeof(float));

        for(int j = 0; j < LSH_level; j++){
		    for(int i = 0; i < m; i++){
	            output.write((char*)(LSH_vec[j][i]), sizeof(float) * LSH_vecdim0);
		    }
		}		

	
        for(int j = 0; j < level; j++){
	        output.write((char*)(id_arr[j]), sizeof(int) * vecdim0);
		}	
	
	    for(int j = 0; j < thres_num; j++)
		output.write((char*)(thres_pos[j]), sizeof(float) * step);		
	
		output.close();
		printf("Indexing finished\n");
		exit(0);
    }


    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    size_t k = topk_;
    //cout << "Parsing gt:\n";
    get_gt(massQA, massQ, mass, vecsize, qsize, l2space, vecdim, answers, k);
    //cout << "Loaded gt\n";

	
	float* query_rot = new float[vecdim];
	
	float** query_pq = new float*[level];
	for(int i = 0; i < level; i++) query_pq[i] = new float[L];

	float** query_lsh = new float*[LSH_level];
	for(int i = 0; i < LSH_level; i++) query_lsh[i] = new float[2 * m];   	
	
	int eps = (int)(eps_ * 20 - 1);

    for (int i = 0; i < 1; i++)
        test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k, LSH_vec, thres_pos[eps], query_rot, query_lsh, permutation, table_size);
    return;


}
