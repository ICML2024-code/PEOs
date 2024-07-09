#include <stdlib.h>

void PEOs(int efc, int M, int vecsize, int qsize, int dim, int table_size, char* path_q_, char* path_data_, char* path_truth_, int L, float eps, int topk);
int main(int argc, char** argv) {
    char* data_path = argv[1];
	char* query_path = argv[2];
	char* truth_path = argv[3];
	int vecsize = atoi(argv[4]);
	int qsize = atoi(argv[5]);
	int dim = atoi(argv[6]);
	int topk = atoi(argv[7]);
	
	int efc = atoi(argv[8]);
	int M = atoi(argv[9]);
	int L = atoi(argv[10]);
    float eps = atof(argv[11]);
    int table_size = atoi(argv[12]);	

    PEOs(efc, M, vecsize, qsize, dim, table_size, query_path, data_path, truth_path, L, eps, topk);

    return 0;
};
