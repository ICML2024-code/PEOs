#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>

//#include <boost/random.hpp>
//#include <boost/random/normal_distribution.hpp>

#define max_edge 64
#define L 256
#define size_n 100000

struct Elem{
  unsigned int id;
  float dist;
};

int Elem_comp2(				// compare function for qsort
	const void* e1,						// 1st element
	const void* e2)						// 2nd element
{
	int ret = 0;
	Elem* value1 = (Elem *) e1;
	Elem* value2 = (Elem *) e2;
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

        struct Neighbor {
            unsigned id;
            float distance;
			float ip;
            bool flag;
	
            Neighbor() = default;
            Neighbor(unsigned id, float distance, float ip, bool f) : id{id}, distance{distance}, ip{ip}, flag(f) {}

            inline bool operator<(const Neighbor &other) const {
                return distance < other.distance;
            }
        };

        static inline int InsertIntoPool (Neighbor *addr, unsigned K, Neighbor nn) {
		
            int left=0,right=K-1;
            if(addr[left].distance>nn.distance){
                memmove((char *)&addr[left+1], &addr[left],K * sizeof(Neighbor));
                addr[left] = nn;
                return left;
            }
            if(addr[right].distance<nn.distance){
                addr[K] = nn;
                return K;
            }
            while(left<right-1){
                int mid=(left+right)/2;
                if(addr[mid].distance>nn.distance)right=mid;
                else left=mid;
            }
 
            while (left > 0){
                if (addr[left].distance < nn.distance) break;
                if (addr[left].id == nn.id) return K + 1;
                left--;
            }
            if(addr[left].id == nn.id||addr[right].id==nn.id)return K+1;
            memmove((char *)&addr[right+1], &addr[right],(K-right) * sizeof(Neighbor));
            addr[right]=nn;
            return right;
        }

namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:
        static const tableint max_update_element_locks = 65536;
        HierarchicalNSW(SpaceInterface<dist_t> *s) {
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, SpaceInterface<dist_t> *s1, SpaceInterface<dist_t> *s2, SpaceInterface<dist_t> *s3, const std::string &location, float diff3, int LSH_level, int LSH_vecdim0, bool nmslib = false, size_t max_elements=0) {
            loadIndex(location, s, s1, s2, s3, diff3, LSH_level, LSH_vecdim0, max_elements);
        }

        HierarchicalNSW(size_t m, float diff, int level, int vecdim0, int LSH_level, int LSH_vecdim0, SpaceInterface<dist_t> *s, SpaceInterface<dist_t> *s1, SpaceInterface<dist_t> *s2, SpaceInterface<dist_t> *s3, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100) :
                link_list_locks_(max_elements), link_list_update_locks_(max_update_element_locks), element_levels_(max_elements) {
            max_elements_ = max_elements;

            num_deleted_ = 0;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
			fstipfunc_ = s2->get_dist_func();		
            dist_func_param_ = s->get_dist_func_param();
			
			fstPQfunc_ = s1->get_dist_func();
			subip_func_param_ = s1->get_dist_func_param();
			fstLSHfunc_ = s3->get_dist_func();
			LSHip_func_param_ = s3->get_dist_func_param();
			
			diff_ = diff;
			
            level_ = level;
            vecdim0_ = vecdim0;

            LSH_level_ = LSH_level;
            LSH_vecdim0_ = LSH_vecdim0;			
			
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction,M_);
            ef_ = 10;
			
			m_ = m;
			
            len_ =  16 * (3 * sizeof(unsigned short int));
			len1_ =  16 * ( sizeof(unsigned short int) );
			len2_ =  16 * (2 * sizeof(unsigned short int) );

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

            size_links_level0_ = maxM0_ * (4 * sizeof(unsigned short int) + sizeof(tableint) + LSH_level_) + sizeof(linklistsizeint);   //new
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype) + sizeof(float); 
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_ + sizeof(float);
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            //initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            size_links_per_element_ = maxM_ * (4 * sizeof(unsigned short int) + sizeof(tableint) + LSH_level_) + sizeof(linklistsizeint);
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;
        }

        struct CompareByFirst {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        ~HierarchicalNSW() {

            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }

        size_t max_elements_;
        size_t cur_element_count;
        size_t size_data_per_element_;
        size_t size_links_per_element_;
        size_t num_deleted_;

        float min_norm_;
		float diff_;
		
		float min_norm2_;
		float diff2_;
		
		int* inverse_id;
		
		float diff3_;
		
		float diffadj_;
		float diffres_;
		float difflast_;
		
        int level_;
		int LSH_level_;
        int vecdim0_;
        int LSH_vecdim0_;		

        size_t len_;
		size_t len1_;
		size_t len2_;
		size_t len3_;
		size_t len4_;
		
		int shift_;

        size_t M_;
        size_t m_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;

        double mult_, revSize_;
        int maxlevel_;


        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;

        std::vector<std::mutex> link_list_locks_;

        // Locks to prevent race condition during update/insert of an element at same time.
        // Note: Locks for additions can also be used to prevent this race condition if the querying of KNN is not exposed along with update/inserts i.e multithread insert/update/query in parallel.
        std::vector<std::mutex> link_list_update_locks_;
        tableint enterpoint_node_;

        size_t size_links_level0_;
        size_t offsetData_, offsetLevel0_;

        //--------new parameters--------------
        size_t size_data_per_element16_;
		size_t size_data_per_element32_;
		size_t size_data_per_element48_;
		
		//size_t label_offset16_;
		//size_t label_offset32_;
		
		size_t sec_part_;
		
		size_t size_links_level16_;
		size_t size_links_level32_;
		size_t size_links_level48_;
		
		size_t num16;
        size_t num32;
        size_t num48;  		
		size_t num64;
		
		char* data_level0_memory32_;
		char* data_level0_memory48_;
		char* data_level0_memory64_;

       //---------------------------------------

        char *data_level0_memory_;
		char *vec_level0_memory_;
        char **linkLists_;
        std::vector<int> element_levels_;

        size_t data_size_;

        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
		DISTFUNC<dist_t> fstipfunc_;
		DISTFUNC<dist_t> fstPQfunc_;
		DISTFUNC<dist_t> fstLSHfunc_;
		
		//size_t m;
		
        void *dist_func_param_;
		void *subip_func_param_;
		void *LSHip_func_param_;
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        inline labeltype getExternalLabel(tableint internal_id) const {
            labeltype return_label;
            memcpy(&return_label,(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const {
            return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        inline int* getNeighborid(int* data, int i) const {
			int* data2 = data + i;
			return data2;
        }

        inline unsigned short int* getFirstNorm(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += maxM0_ * (sizeof(tableint)) + a * len_ + b * sizeof(unsigned short int);
			return (unsigned short int* )data2;
        }
		
        inline unsigned short int* getFirstNorm16(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += 16 * (sizeof(tableint)) + a * len_ + b * sizeof(unsigned short int);
			return (unsigned short int* )data2;
        }

        inline unsigned short int* getFirstNorm32(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += maxM_ * (sizeof(tableint)) + a * len_ + b * sizeof(unsigned short int);
			return (unsigned short int* )data2;
        }

        inline unsigned short int* getFirstNorm48(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += 48 * (sizeof(tableint)) + a * len_ + b * sizeof(unsigned short int);
			return (unsigned short int* )data2;
        }		
		
        inline short int* getAdjustNorm(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += maxM0_ * (sizeof(tableint)) + a * len_ + len1_ +  b * sizeof(unsigned short int);
			return (short int* )data2;
        }
		
        inline short int* getAdjustNorm16(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += 16 * (sizeof(tableint)) + a * len_ + len1_ +  b * sizeof(unsigned short int);
			return (short int* )data2;
        }

        inline short int* getAdjustNorm32(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += maxM_ * (sizeof(tableint)) + a * len_ + len1_ +  b * sizeof(unsigned short int);
			return (short int* )data2;
        }
		
		inline short int* getAdjustNorm48(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += 48 * (sizeof(tableint)) + a * len_ + len1_ +  b * sizeof(unsigned short int);
			return (short int* )data2;
        }
			
        inline unsigned short int* getLSHNorm(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += maxM0_ * (sizeof(tableint)) + a * len_ + len2_ + b * sizeof(unsigned short int);
			return (unsigned short int* )data2;
        }
		
        inline unsigned short int* getLSHNorm16(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += 16 * (sizeof(tableint)) + a * len_ + len2_ + b * sizeof(unsigned short int);
			return (unsigned short int* )data2;
        }

        inline unsigned short int* getLSHNorm32(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += maxM_ * (sizeof(tableint)) + a * len_ + len2_ + b * sizeof(unsigned short int);
			return (unsigned short int* )data2;
        }

        inline unsigned short int* getLSHNorm48(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += 48 * (sizeof(tableint)) + a * len_ + len2_ + b * sizeof(unsigned short int);
			return (unsigned short int* )data2;
        }

        inline short int* getLastNorm(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += maxM0_ * (sizeof(tableint) + 3 * sizeof(unsigned short int)) + a * len1_ + b * sizeof(unsigned short int);
			return (short int* )data2;
        }
		
        inline short int* getLastNorm16(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += 16 * (sizeof(tableint) + 3 * sizeof(unsigned short int)) + a * len1_ + b * sizeof(unsigned short int);
			return (short int* )data2;
        }

        inline short int* getLastNorm32(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += maxM_ * (sizeof(tableint) + 3 * sizeof(unsigned short int)) + a * len1_ + b * sizeof(unsigned short int);
			return (short int* )data2;
        }		

        inline short int* getLastNorm48(int* data, int i) const {
			int a = i / 16;
			int b = i % 16; 			
			char* data2 = (char*) data;
			data2 += 48 * (sizeof(tableint) + 3 * sizeof(unsigned short int)) + a * len1_ + b * sizeof(unsigned short int);
			return (short int* )data2;
        }

        inline unsigned char* getLSHM(int* data, int level, int i) const {
			char* data2 = (char*) data;
			data2 += maxM0_ * (sizeof(int) + 4 * sizeof(unsigned short int) + level) + i;
			return (unsigned char* )data2;
            //return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        inline unsigned char* getLSHM16(int* data, int level, int i) const {
			char* data2 = (char*) data;
			data2 += 16 * (sizeof(int) + 4 * sizeof(unsigned short int) + level) + i;
			return (unsigned char* )data2;
            //return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        inline unsigned char* getLSHM32(int* data, int level, int i) const {
			char* data2 = (char*) data;
			data2 += maxM_ * (sizeof(int) + 4 * sizeof(unsigned short int) + level) + i;
			return (unsigned char* )data2;
            //return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }		

        inline unsigned char* getLSHM48(int* data, int level, int i) const {
			char* data2 = (char*) data;
			data2 += 48 * (sizeof(int) + 4 * sizeof(unsigned short int) + level) + i;
			return (unsigned char* )data2;
            //return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }
		
        inline char *getNormByInternalIdQuery(tableint internal_id) const {
            return (vec_level0_memory_ + internal_id * sec_part_);
        }

        inline labeltype getExternalLabelQuery(tableint internal_id) const {
            labeltype return_label;
            memcpy(&return_label,(vec_level0_memory_ + internal_id * sec_part_ + label_offset_), sizeof(labeltype));
            return return_label;
        }	

        inline char *getNormByInternalId(tableint internal_id) const {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }
		
        inline char *getDataByInternalId(tableint internal_id) const {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_ + sizeof(float));
        }

        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int) r;
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0) {
                    data = (int*)get_linklist0(curNodeNum);
                } else {
                    data = (int*)get_linklist(curNodeNum, layer);
                }
                size_t size = getListCount((linklistsizeint*)data);
                tableint *datal = (tableint *) (data + 1);
				
				int* data2 = getNeighborid((int*) datal, 0);
				int* data3 = getNeighborid((int*) datal, 1);
				
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data2)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data2) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*data2), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(data3)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
					int candidate_id = *(getNeighborid((int*)datal, j));
                    //tableint candidate_id = *(datal + j);
					int* data4 = getNeighborid((int*)datal, j);
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data4)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(data4)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        mutable std::atomic<long> metric_distance_computations;
        mutable std::atomic<long> metric_hops;

        template <bool has_deletions, bool collect_metrics=false>
        void searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef, unsigned int* result, int K, float*** lsh_vec, float* thres_pos, float** query_lsh, int table_size) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            //-------------------------------------
			int LL = ef;
            std::vector<Neighbor> retset(LL + 1);
            
			int vecdim = LSH_vecdim0_ * LSH_level_;	            		
			
			float square_q = fstipfunc_(data_point, data_point, dist_func_param_);
	
			float* x = (float *) data_point;
			    for(int j = 0; j < LSH_level_; j++){
				    float* y = x + j * LSH_vecdim0_;
                    for(int i = 0; i < m_; i++){
				        _mm_prefetch((char *) (lsh_vec[j][i]), _MM_HINT_T0);
				        query_lsh[j][i] = fstLSHfunc_( (void*) y, (void*) lsh_vec[j][i], LSHip_func_param_);
                        query_lsh[j][i + m_] = -1.0f * query_lsh[j][i];
                    }
			    }	
				
			char* norm_pointer0 = getNormByInternalIdQuery(ep_id);
			float true_norm0 = *((float*) norm_pointer0); 
			norm_pointer0 += 4;
			float ip0 = fstipfunc_(data_point, norm_pointer0, dist_func_param_);
			float dist0 = true_norm0 * ( true_norm0 * 0.5 - ip0);
             
			
		
            retset[0] = Neighbor(ep_id, dist0, ip0, true);
            visited_array[ep_id] = visited_array_tag; 

            int k = 0;
			int l_num = 1;
			
            float PORTABLE_ALIGN64 Thres1[16];	
            float PORTABLE_ALIGN64 Thres2[16];
			float PORTABLE_ALIGN64 Thres3[16];
			float PORTABLE_ALIGN64 Thres4[16];
			
			int* real_data = new int[maxM0_];
		
	        float ss0 = diff3_ / table_size;
	
            while (k < LL) {
                int nk = LL;

                if (retset[k].flag) {
                    retset[k].flag = false;
                    unsigned n = retset[k].id;

					if(l_num < LL){	
                        int *data; 
						
						if(n < num16){
						    data = (int *) get_linklist16(n);
						}
						else if(n < num32){
							data = (int *) get_linklist32(n - num16);
						}
						else if(n < num48){
							data = (int *) get_linklist48(n - num32);
						}
						else{
							data = (int *) get_linklist64(n - num48);
						}
					
                        size_t size = getListCount((linklistsizeint*)data);
						int* datal = data + 1;
				        int* data2 = getNeighborid(datal, 0);
						int* data3 = getNeighborid(datal, 1);		

#ifdef USE_SSE
                        _mm_prefetch((char *) (visited_array + *data2), _MM_HINT_T0);
                        _mm_prefetch((char *) (visited_array + *data2 + 64), _MM_HINT_T0);
                        _mm_prefetch(vec_level0_memory_ + (*data2) * sec_part_, _MM_HINT_T0);
                        _mm_prefetch((char *) (data3), _MM_HINT_T0);						
						
#endif

                        for (size_t j = 1; j <= size; j++) {					
							int candidate_id = *(getNeighborid(datal, j-1));
							int* data4 = getNeighborid(datal, j);
							             
#ifdef USE_SSE
                            _mm_prefetch((char *) (visited_array + *data4), _MM_HINT_T0);
                            _mm_prefetch(vec_level0_memory_ + (*data4) * sec_part_,
                            _MM_HINT_T0);////////////

#endif
                            if (!(visited_array[candidate_id] == visited_array_tag)) {
                        
                                visited_array[candidate_id] = visited_array_tag;
	
			                    char* norm_pointer = getNormByInternalIdQuery(candidate_id);
			                    float true_norm = *((float*) norm_pointer); 
			                    norm_pointer += 4;
			                    float ip_ = fstipfunc_(data_point, norm_pointer, dist_func_param_);
			                    float dist = true_norm * ( true_norm * 0.5 - ip_);
								
                                if (l_num == LL && dist >= retset[LL - 1].distance ) continue;

                                int r;
		                        if(l_num == LL){
 			
                 		            Neighbor nn2(candidate_id, dist, ip_, true);
                                    r = InsertIntoPool(retset.data(), LL, nn2);
		                        }
	                 	        else {
                                    Neighbor nn(candidate_id, dist, ip_, true);
                                    r = InsertIntoPool(retset.data(), l_num, nn);
			                        l_num++;
                                }
                                if (r < nk) {nk = r;}
						    }							
					    }
					}
					else{
						int* data;
						if(n < num16){
						    data = (int *) get_linklist16(n);
						}
						else if(n < num32){
							data = (int *) get_linklist32(n - num16);
						}
						else if(n < num48){
							data = (int *) get_linklist48(n - num32);
						}
						else{
							data = (int *) get_linklist64(n - num48);
						}						
						int* datal = data + 1;
						size_t size = getListCount((linklistsizeint*)data);
						
						float val = retset[LL-1].distance; 
						float qcosine = retset[k].ip;
						float Lbound = qcosine * diffadj_;
						//float res_val = qcosine * diffres_;
						
						float qsine = sqrt(square_q - (qcosine * qcosine));
						float ss = qsine * ss0;						
						
						float coff2 = qcosine * difflast_;
						
                        int count = 0;            
                        int LB = 1;
                        int UB = table_size - 1;
                        int step1 = table_size - 1;						
						
//-------------------------------estimation part--------------------------------
                        //--------------------------------------------------
						
						int div = size % 16;
						int round;
						if(div == 0){
							round = size / 16;
							div = 16;
						}else{
							round = size / 16 + 1;
						}							
							
						__m512i v_int;
						__m256i v_short;
                        __m128i v_char;								
                        __m512 v_float, temp_sub, temp_center;
						temp_sub = _mm512_set1_ps(val);
						temp_center = _mm512_set1_ps(Lbound);
						//temp_res  = _mm512_set1_ps(res_val);
                        __m512 sum1, sum2, sum3, sum4;

			            __m512 diff_1, diff_3;								
			            diff_1 = _mm512_set1_ps(diff_);
                        diff_3 = _mm512_set1_ps(ss);
                    if(round == 1){
						unsigned short int* short_pointer = (unsigned short int*) (datal + 16);

						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//1st
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						sum1 = _mm512_mul_ps(v_float, diff_1);
						sum1 = _mm512_sub_ps(sum1, temp_sub);
	
						short_pointer += 16;
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//adjust norm
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, temp_center);
						sum1 = _mm512_sub_ps(sum1, v_float);
						
						short_pointer += 16;   					
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//3rd
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, diff_3);	
                        sum1 = _mm512_div_ps(sum1, v_float);
						short_pointer += 16;
												
						//_mm512_store_ps(Thres1, sum1);
	//-------------------------LSH part---------------------------------------	
						//-------------compute LSH code---------------
						_mm_prefetch((char *) (thres_pos), _MM_HINT_T0);						
						
						v_int = _mm512_cvtps_epi32(sum1);
						__mmask16 k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(LB), _MM_CMPINT_LE);
						__m512i neg = _mm512_setzero_epi32();
						
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(UB), _MM_CMPINT_GE);
						neg = _mm512_set1_epi32(step1);
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						sum1 = _mm512_i32gather_ps(v_int, thres_pos, 4);//

                        __m512 v_tmp = _mm512_set1_ps(qsine);  //new1
						sum1 = _mm512_mul_ps(sum1, v_tmp);
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//last
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_tmp = _mm512_set1_ps(coff2);
						v_float = _mm512_mul_ps(v_float, v_tmp);
                        sum1 = _mm512_add_ps(sum1, v_float);						
						short_pointer += 16;
					
						//---------------------------------------------------------
						//-------------compute LSH code---------------
                                
						unsigned char* char_pointer = (unsigned char*) short_pointer;
						
						//-------------LSH first codebook-----------------------
					
                        for(int i = 0; i < LSH_level_; i++){
                            float* lsh_pointer = query_lsh[i];


						    _mm_prefetch((char *) (lsh_pointer), _MM_HINT_T0);
						    _mm_prefetch((char *) (lsh_pointer + 64), _MM_HINT_T0);
						    _mm_prefetch((char *) (lsh_pointer + 128), _MM_HINT_T0);
						    _mm_prefetch((char *) (lsh_pointer + 192), _MM_HINT_T0);


									
                            v_char = _mm_loadu_si128((__m128i*) char_pointer); 
							v_int = _mm512_cvtepu8_epi32(v_char);
					        v_float = _mm512_i32gather_ps(v_int, lsh_pointer, 4);
						    sum1 = _mm512_sub_ps(sum1, v_float);
                            char_pointer += 16;
									
						}
						
                        //---------------------------------------------------

						for(int i = 0; i < div; i++){
							if(Thres1[i] <= 0){
                                real_data[count] = datal[i];
								count++;
							}
						}
						
					}
                    else if(round == 2){
                        unsigned short int* short_pointer = (unsigned short int*) (datal + 32);
									
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//1st
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						sum1 = _mm512_mul_ps(v_float, diff_1);
						sum1 = _mm512_sub_ps(sum1, temp_sub);
	
						short_pointer += 16;
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//adjust norm
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, temp_center);
						sum1 = _mm512_sub_ps(sum1, v_float);
						
						short_pointer += 16;   					
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//3rd
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, diff_3);	
                        sum1 = _mm512_div_ps(sum1, v_float);
						short_pointer += 16;
						
						//_mm512_store_ps(Thres1, sum1);
	//----------------------------------------------------------------					
//----------------------------------------------------------
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//1st
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						sum2 = _mm512_mul_ps(v_float, diff_1);
						sum2 = _mm512_sub_ps(sum2, temp_sub);	
	
						short_pointer += 16;
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//adjust norm
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, temp_center);
						sum2 = _mm512_sub_ps(sum2, v_float);
						
						short_pointer += 16;   					
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//3rd
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, diff_3);	
                        sum2 = _mm512_div_ps(sum2, v_float);
						short_pointer += 16;
						
						//_mm512_store_ps(Thres2, sum2);
	//----------------------------------------------------------------
						//-------------compute LSH code---------------
						_mm_prefetch((char *) (thres_pos), _MM_HINT_T0);						
						
						v_int = _mm512_cvtps_epi32(sum1);
						__mmask16 k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(LB), _MM_CMPINT_LE);
						__m512i neg = _mm512_setzero_epi32();
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(UB), _MM_CMPINT_GE);
						neg = _mm512_set1_epi32(step1);
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						sum1 = _mm512_i32gather_ps(v_int, thres_pos, 4);//----
						
                        __m512 v_tmp = _mm512_set1_ps(qsine);   //new1
						sum1 = _mm512_mul_ps(sum1, v_tmp);
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//last
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_tmp = _mm512_set1_ps(coff2);
						v_float = _mm512_mul_ps(v_float, v_tmp);
                        sum1 = _mm512_add_ps(sum1, v_float);						
						short_pointer += 16;						

						v_int = _mm512_cvtps_epi32(sum2);
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(LB), _MM_CMPINT_LE);
						neg = _mm512_setzero_epi32();
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(UB), _MM_CMPINT_GE);
						neg = _mm512_set1_epi32(step1);
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						sum2 = _mm512_i32gather_ps(v_int, thres_pos, 4);  //
	
                        v_tmp = _mm512_set1_ps(qsine);   //new1
						sum2 = _mm512_mul_ps(sum2, v_tmp);
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//last
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_tmp = _mm512_set1_ps(coff2);
						v_float = _mm512_mul_ps(v_float, v_tmp);
                        sum2 = _mm512_add_ps(sum2, v_float);						
						short_pointer += 16;	
						//v_float = _mm512_set1_epi32(qsine);
						//sum2 = 
					
						//---------------------------------------------------------
						//-------------compute LSH code---------------
                                
						unsigned char* char_pointer = (unsigned char*) short_pointer;
						
						//-------------LSH codebook----------						
                        for(int i = 0; i < LSH_level_; i++){
                            float* lsh_pointer = query_lsh[i];             

						    _mm_prefetch((char *) (lsh_pointer), _MM_HINT_T0);
						    _mm_prefetch((char *) (lsh_pointer + 64), _MM_HINT_T0);
						    _mm_prefetch((char *) (lsh_pointer + 128), _MM_HINT_T0);
						    _mm_prefetch((char *) (lsh_pointer + 192), _MM_HINT_T0);
									
                            v_char = _mm_loadu_si128((__m128i*) char_pointer); 
							v_int = _mm512_cvtepu8_epi32(v_char);
					        v_float = _mm512_i32gather_ps(v_int, lsh_pointer, 4);
						    sum1 = _mm512_sub_ps(sum1, v_float);
                            char_pointer += 16;

                            v_char = _mm_loadu_si128((__m128i*) char_pointer); 
							v_int = _mm512_cvtepu8_epi32(v_char);
					        v_float = _mm512_i32gather_ps(v_int, lsh_pointer, 4);
						    sum2 = _mm512_sub_ps(sum2, v_float);
                            char_pointer += 16; 									
						}
								
						_mm512_store_ps(Thres1, sum1);
                        _mm512_store_ps(Thres2, sum2);
					
						//---------------------------------------------------	

	
						//bool* check_pointer = is_checked;		
						for(int i = 0; i < 16; i++){
							if(Thres1[i] <= 0){
                                real_data[count] = datal[i];
								count++;
							}
						}
						
						int* datall = datal + 16;
						for(int i = 0; i < div; i++){
							if(Thres2[i] <= 0){
                                real_data[count] = datall[i];
								count++;
							}

						}						
					}

                    else if(round == 3){
                        unsigned short int* short_pointer = (unsigned short int*) (datal + 48);

						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//1st
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						sum1 = _mm512_mul_ps(v_float, diff_1);
						sum1 = _mm512_sub_ps(sum1, temp_sub);	
	
						short_pointer += 16;
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//adjust norm
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, temp_center);
						sum1 = _mm512_sub_ps(sum1, v_float);
						
						short_pointer += 16;   					
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//3rd
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, diff_3);	
                        sum1 = _mm512_div_ps(sum1, v_float);
						short_pointer += 16;
						
						//_mm512_store_ps(Thres1, sum1);
	//----------------------------------------------------------------					
//----------------------------------------------------------
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//1st
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						sum2 = _mm512_mul_ps(v_float, diff_1);
						sum2 = _mm512_sub_ps(sum2, temp_sub);	
	
						short_pointer += 16;
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//adjust norm
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, temp_center);
						sum2 = _mm512_sub_ps(sum2, v_float);
						
						short_pointer += 16;   					
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//3rd
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, diff_3);	
                        sum2 = _mm512_div_ps(sum2, v_float);
						short_pointer += 16;
						
						//_mm512_store_ps(Thres2, sum2);
	//----------------------------------------------------------------
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//1st
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						sum3 = _mm512_mul_ps(v_float, diff_1);
						sum3 = _mm512_sub_ps(sum3, temp_sub);
						
						short_pointer += 16;
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//adjust norm
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, temp_center);
						sum3 = _mm512_sub_ps(sum3, v_float);
						
						short_pointer += 16;   					
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//3rd
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, diff_3);	
                        sum3 = _mm512_div_ps(sum3, v_float);
						short_pointer += 16;
						
						//_mm512_store_ps(Thres3, sum3);

						//-------------compute LSH code---------------
						_mm_prefetch((char *) (thres_pos), _MM_HINT_T0);						
						
						v_int = _mm512_cvtps_epi32(sum1);
						__mmask16 k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(LB), _MM_CMPINT_LE);
						__m512i neg = _mm512_setzero_epi32();
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(UB), _MM_CMPINT_GE);
						neg = _mm512_set1_epi32(step1);
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						sum1 = _mm512_i32gather_ps(v_int, thres_pos, 4);//

                        __m512 v_tmp = _mm512_set1_ps(qsine);   //new1
						sum1 = _mm512_mul_ps(sum1, v_tmp);
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//last
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_tmp = _mm512_set1_ps(coff2);
						v_float = _mm512_mul_ps(v_float, v_tmp);
                        sum1 = _mm512_add_ps(sum1, v_float);						
						short_pointer += 16;	

						v_int = _mm512_cvtps_epi32(sum2);
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(LB), _MM_CMPINT_LE);
						neg = _mm512_setzero_epi32();
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(UB), _MM_CMPINT_GE);
						neg = _mm512_set1_epi32(step1);
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						sum2 = _mm512_i32gather_ps(v_int, thres_pos, 4);  //

                        v_tmp = _mm512_set1_ps(qsine);   //new1
						sum2 = _mm512_mul_ps(sum2, v_tmp);
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//last
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_tmp = _mm512_set1_ps(coff2);
						v_float = _mm512_mul_ps(v_float, v_tmp);
                        sum2 = _mm512_add_ps(sum2, v_float);						
						short_pointer += 16;	

						v_int = _mm512_cvtps_epi32(sum3);
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(LB), _MM_CMPINT_LE);
						neg = _mm512_setzero_epi32();
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(UB), _MM_CMPINT_GE);
						neg = _mm512_set1_epi32(step1);
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);

						sum3 = _mm512_i32gather_ps(v_int, thres_pos, 4); //

                        v_tmp = _mm512_set1_ps(qsine);   //new1
						sum3 = _mm512_mul_ps(sum3, v_tmp);
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//last
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_tmp = _mm512_set1_ps(coff2);
						v_float = _mm512_mul_ps(v_float, v_tmp);
                        sum3 = _mm512_add_ps(sum3, v_float);						
						short_pointer += 16;	
					
						//---------------------------------------------------------
						//-------------compute LSH code---------------
                                
						unsigned char* char_pointer = (unsigned char*) short_pointer;
						
                        for(int i = 0; i < LSH_level_; i++){
                            float* lsh_pointer = query_lsh[i];

						    _mm_prefetch((char *) (lsh_pointer), _MM_HINT_T0);
						    _mm_prefetch((char *) (lsh_pointer + 64), _MM_HINT_T0);
						    _mm_prefetch((char *) (lsh_pointer + 128), _MM_HINT_T0);
						    _mm_prefetch((char *) (lsh_pointer + 192), _MM_HINT_T0);
									
                            v_char = _mm_loadu_si128((__m128i*) char_pointer); 
							v_int = _mm512_cvtepu8_epi32(v_char);
					        v_float = _mm512_i32gather_ps(v_int, lsh_pointer, 4);
						    sum1 = _mm512_sub_ps(sum1, v_float);
                            char_pointer += 16;

                            v_char = _mm_loadu_si128((__m128i*) char_pointer); 
							v_int = _mm512_cvtepu8_epi32(v_char);
					        v_float = _mm512_i32gather_ps(v_int, lsh_pointer, 4);
						    sum2 = _mm512_sub_ps(sum2, v_float);
                            char_pointer += 16; 

                            v_char = _mm_loadu_si128((__m128i*) char_pointer); 
							v_int = _mm512_cvtepu8_epi32(v_char);
					        v_float = _mm512_i32gather_ps(v_int, lsh_pointer, 4);
						    sum3 = _mm512_sub_ps(sum3, v_float);
                            char_pointer += 16; 									
						}
								
						_mm512_store_ps(Thres1, sum1);
                        _mm512_store_ps(Thres2, sum2);
                        _mm512_store_ps(Thres3, sum3);
						
						//bool* check_pointer = is_checked;		
						for(int i = 0; i < 16; i++){
							if(Thres1[i] <= 0){
	                            real_data[count] = datal[i];
								count++;
							}
						}
						
						int* datall = datal + 16;
						for(int i = 0; i < 16; i++){
							if(Thres2[i] <= 0){
                                real_data[count] = datall[i];
								count++;
							}
						}

                        datall += 16;
						for(int i = 0; i < div; i++){
							if(Thres3[i] <= 0){
                                real_data[count] = datall[i];
								count++;
							}
						}
						
					}

                    else if(round == 4){
                        unsigned short int* short_pointer = (unsigned short int*) (datal + maxM0_);
						
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//1st
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						sum1 = _mm512_mul_ps(v_float, diff_1);
						sum1 = _mm512_sub_ps(sum1, temp_sub);	
	
						short_pointer += 16;
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//adjust norm
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, temp_center);
						sum1 = _mm512_sub_ps(sum1, v_float);
						
						short_pointer += 16;   					
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//3rd
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, diff_3);	
                        sum1 = _mm512_div_ps(sum1, v_float);
						short_pointer += 16;
						
						//_mm512_store_ps(Thres1, sum1);
	//----------------------------------------------------------------					
//----------------------------------------------------------
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//1st
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						sum2 = _mm512_mul_ps(v_float, diff_1);
						sum2 = _mm512_sub_ps(sum2, temp_sub);		
	
						short_pointer += 16;
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//adjust norm
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, temp_center);
						sum2 = _mm512_sub_ps(sum2, v_float);
						
						short_pointer += 16;   					
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//3rd
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, diff_3);	
                        sum2 = _mm512_div_ps(sum2, v_float);
						short_pointer += 16;
						
						//_mm512_store_ps(Thres2, sum2);
	//----------------------------------------------------------------
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//1st
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						sum3 = _mm512_mul_ps(v_float, diff_1);
						sum3 = _mm512_sub_ps(sum3, temp_sub);		
	
						short_pointer += 16;
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//adjust norm
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, temp_center);
						sum3 = _mm512_sub_ps(sum3, v_float);
						
						short_pointer += 16;   					
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//3rd
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, diff_3);	
                        sum3 = _mm512_div_ps(sum3, v_float);
						short_pointer += 16;
						
						//_mm512_store_ps(Thres3, sum3);
//-----------------------------------------------------------------------------------
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//1st
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						sum4 = _mm512_mul_ps(v_float, diff_1);
						sum4 = _mm512_sub_ps(sum4, temp_sub);	
	
						short_pointer += 16;
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//adjust norm
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, temp_center);
						sum4 = _mm512_sub_ps(sum4, v_float);
						
						short_pointer += 16;   					
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//3rd
						v_int = _mm512_cvtepu16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_float = _mm512_mul_ps(v_float, diff_3);	
                        sum4 = _mm512_div_ps(sum4, v_float);
						short_pointer += 16;
						
						//_mm512_store_ps(Thres4, sum4);					
						//-----------mask part---------------------------------
						_mm_prefetch((char *) (thres_pos), _MM_HINT_T0);						
						
						v_int = _mm512_cvtps_epi32(sum1);
						__mmask16 k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(LB), _MM_CMPINT_LE);
						__m512i neg = _mm512_setzero_epi32();
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(UB), _MM_CMPINT_GE);
						neg = _mm512_set1_epi32(step1);
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						sum1 = _mm512_i32gather_ps(v_int, thres_pos, 4);//
						
                        __m512 v_tmp = _mm512_set1_ps(qsine);   //new1
						sum1 = _mm512_mul_ps(sum1, v_tmp);
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//last
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_tmp = _mm512_set1_ps(coff2);
						v_float = _mm512_mul_ps(v_float, v_tmp);
                        sum1 = _mm512_add_ps(sum1, v_float);						
						short_pointer += 16;						

						v_int = _mm512_cvtps_epi32(sum2);
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(LB), _MM_CMPINT_LE);
						neg = _mm512_setzero_epi32();
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(UB), _MM_CMPINT_GE);
						neg = _mm512_set1_epi32(step1);
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						sum2 = _mm512_i32gather_ps(v_int, thres_pos, 4);  //
						
                        v_tmp = _mm512_set1_ps(qsine);   //new1
						sum2 = _mm512_mul_ps(sum2, v_tmp);
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//last
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_tmp = _mm512_set1_ps(coff2);
						v_float = _mm512_mul_ps(v_float, v_tmp);
                        sum2 = _mm512_add_ps(sum2, v_float);						
						short_pointer += 16;						

						v_int = _mm512_cvtps_epi32(sum3);
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(LB), _MM_CMPINT_LE);
						neg = _mm512_setzero_epi32();
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(UB), _MM_CMPINT_GE);
						neg = _mm512_set1_epi32(step1);
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);

						sum3 = _mm512_i32gather_ps(v_int, thres_pos, 4); //
						
                        v_tmp = _mm512_set1_ps(qsine);   //new1
						sum3 = _mm512_mul_ps(sum3, v_tmp);
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//last
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_tmp = _mm512_set1_ps(coff2);
						v_float = _mm512_mul_ps(v_float, v_tmp);
                        sum3 = _mm512_add_ps(sum3, v_float);						
						short_pointer += 16;						

						v_int = _mm512_cvtps_epi32(sum4);
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(LB), _MM_CMPINT_LE);
						neg = _mm512_setzero_epi32();
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						k =_mm512_cmp_epi32_mask(v_int, _mm512_set1_epi32(UB), _MM_CMPINT_GE);
						neg = _mm512_set1_epi32(step1);
						v_int = _mm512_mask_mov_epi32(v_int, k, neg);
						
						sum4 = _mm512_i32gather_ps(v_int, thres_pos, 4);

                        v_tmp = _mm512_set1_ps(qsine);   //new1
						sum4 = _mm512_mul_ps(sum4, v_tmp);
						v_short = _mm256_loadu_si256((__m256i*) short_pointer);//last
						v_int = _mm512_cvtepi16_epi32(v_short);
						v_float = _mm512_cvtepi32_ps(v_int);
						v_tmp = _mm512_set1_ps(coff2);
						v_float = _mm512_mul_ps(v_float, v_tmp);
                        sum4 = _mm512_add_ps(sum4, v_float);						
						short_pointer += 16;						
						//---------------------------------------------------------
						//-------------compute LSH code---------------
                                
						unsigned char* char_pointer = (unsigned char*) short_pointer;
						
						//-------------LSH codebook-----------------------
                        for(int i = 0; i < LSH_level_; i++){
                            float* lsh_pointer = query_lsh[i];

						    _mm_prefetch((char *) (lsh_pointer), _MM_HINT_T0);
						    _mm_prefetch((char *) (lsh_pointer + 64), _MM_HINT_T0);
						    _mm_prefetch((char *) (lsh_pointer + 128), _MM_HINT_T0);
						    _mm_prefetch((char *) (lsh_pointer + 192), _MM_HINT_T0);
									
                            v_char = _mm_loadu_si128((__m128i*) char_pointer); 
							v_int = _mm512_cvtepu8_epi32(v_char);
					        v_float = _mm512_i32gather_ps(v_int, lsh_pointer, 4);
						    sum1 = _mm512_sub_ps(sum1, v_float);
                            char_pointer += 16;

                            v_char = _mm_loadu_si128((__m128i*) char_pointer); 
							v_int = _mm512_cvtepu8_epi32(v_char);
					        v_float = _mm512_i32gather_ps(v_int, lsh_pointer, 4);
						    sum2 = _mm512_sub_ps(sum2, v_float);
                            char_pointer += 16; 

                            v_char = _mm_loadu_si128((__m128i*) char_pointer); 
							v_int = _mm512_cvtepu8_epi32(v_char);
					        v_float = _mm512_i32gather_ps(v_int, lsh_pointer, 4);
						    sum3 = _mm512_sub_ps(sum3, v_float);
                            char_pointer += 16; 

                            v_char = _mm_loadu_si128((__m128i*) char_pointer); 
							v_int = _mm512_cvtepu8_epi32(v_char);
					        v_float = _mm512_i32gather_ps(v_int, lsh_pointer, 4);
						    sum4 = _mm512_sub_ps(sum4, v_float);
                            char_pointer += 16;  									
						}
								
						_mm512_store_ps(Thres1, sum1);
                        _mm512_store_ps(Thres2, sum2);
                        _mm512_store_ps(Thres3, sum3);
                        _mm512_store_ps(Thres4, sum4);						

						
						//bool* check_pointer = is_checked;		
						for(int i = 0; i < 16; i++){
							if(Thres1[i] <= 0){
                                real_data[count] = datal[i];
								count++;
							}
						}
						
						int* datall = datal + 16;
						for(int i = 0; i < 16; i++){
							if(Thres2[i] <= 0){
                                real_data[count] = datall[i];
								count++;
							}
						}

                        datall += 16;
						for(int i = 0; i < 16; i++){
							if(Thres3[i] <= 0){
                                real_data[count] = datall[i];
								count++;
							}
						}

                        datall += 16;
						for(int i = 0; i < div; i++){
							if(Thres4[i] <= 0){
                                real_data[count] = datall[i];
								count++;								
							}
						}						
					}
					
//-----------------------------------------------------------------------------

                        size = count;
						datal = real_data;

						int* data2 = getNeighborid(datal, 0);
						int* data3 = getNeighborid(datal, 1);

#ifdef USE_SSE
                        _mm_prefetch((char *) (visited_array + *data2), _MM_HINT_T0);
                        _mm_prefetch((char *) (visited_array + *data2 + 64), _MM_HINT_T0);
                        _mm_prefetch(vec_level0_memory_+ (*data2) * sec_part_, _MM_HINT_T0);
                        _mm_prefetch((char *) (data3), _MM_HINT_T0);
#endif
                        for (size_t j = 0; j < size; j++) {
							int candidate_id = *getNeighborid(datal, j);
							int* data4 = getNeighborid(datal, j+1);

#ifdef USE_SSE
                            _mm_prefetch((char *) (visited_array + *data4), _MM_HINT_T0);
                            _mm_prefetch(vec_level0_memory_ + (*data4) * sec_part_,
                            _MM_HINT_T0);////////////
#endif							       
                            if (!(visited_array[candidate_id] == visited_array_tag)) {
								visited_array[candidate_id] = visited_array_tag;
								//if(is_checked[j] == false){continue;}
								
			                    char* norm_pointer = getNormByInternalIdQuery(candidate_id);

			                    float true_norm = *((float*) norm_pointer); 
			                    norm_pointer += 4;
			                    float ip_ = fstipfunc_(data_point, norm_pointer, dist_func_param_);
			                    float dist = true_norm * ( true_norm * 0.5 - ip_);
								
                                if (l_num == LL && dist >= retset[LL - 1].distance ) continue;

                                int r;
		                        if(l_num == LL){
 			
                 		            Neighbor nn2(candidate_id, dist, ip_, true);
                                    r = InsertIntoPool(retset.data(), LL, nn2);
		                        }
	                 	        else {
                                    Neighbor nn(candidate_id, dist, ip_, true);
                                    r = InsertIntoPool(retset.data(), l_num, nn);
			                        l_num++;									
                                }
                                if (r < nk) {nk = r;}						
							}												
					    }				
					} 					
				}
				if (nk <= k)
                k = nk;
                else {++k;}
			}			
			
            for (size_t i = 0; i < K; i++) {
                result[i] = getExternalLabelQuery(retset[i].id);
            }			
					
            visited_list_pool_->releaseVisitedList(vl);
		} 			

        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
            if (top_candidates.size() < M) {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list) {
                    dist_t curdist =
                            fstdistfunc_(getDataByInternalId(second_pair.second),
                                         getDataByInternalId(curent_pair.second),
                                         dist_func_param_);;
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }


        linklistsizeint *get_linklist16(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_links_level16_ );
        };
		
		linklistsizeint *get_linklist32(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory32_ + internal_id * size_links_level32_ );
        };

		linklistsizeint *get_linklist48(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory48_ + internal_id * size_links_level48_ );
        };
		
		linklistsizeint *get_linklist64(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory64_ + internal_id * size_links_level0_);
        };



        linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist(tableint internal_id, int level) const {
            return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        };

        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        };

        tableint mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level, bool isUpdate) {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(top_candidates, M_);
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();

            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
				
                setListCount(ll_cur,selectedNeighbors.size());
                
				int *data = (int *) (ll_cur + 1);
				//data += 1;
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
					int* data2 = getNeighborid(data, idx);
                    if (*data2 && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    *data2 = selectedNeighbors[idx];
                }
            }

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                int *data = (int *) (ll_other + 1);
                bool is_cur_c_present = false;
                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
						int* data2 = getNeighborid(data, j);
                        if (*data2 == cur_c) {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) {
                    if (sz_link_list_other < Mcurmax) {
						int* data2 = getNeighborid(data, sz_link_list_other);
                        //data[sz_link_list_other] = cur_c;
						*data2 = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    } else {
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                    dist_func_param_);
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++) {
						    int* data2 = getNeighborid(data, j);
                            candidates.emplace(
                                    fstdistfunc_(getDataByInternalId(*data2), getDataByInternalId(selectedNeighbors[idx]),
                                                 dist_func_param_), *data2);
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax);

                        int indx = 0;
                        while (candidates.size() > 0) {
							int* data2 = getNeighborid(data, indx);
                            *data2 = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }

        std::mutex global;
        size_t ef_;

        void setEf(size_t ef) {
            ef_ = ef;
        }


        std::priority_queue<std::pair<dist_t, tableint>> searchKnnInternal(void *query_data, int k) {
            std::priority_queue<std::pair<dist_t, tableint  >> top_candidates;
			/*
            if (cur_element_count == 0) return top_candidates;
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (size_t level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    int *data;
                    data = (int *) get_linklist(currObj,level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            if (num_deleted_) {
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<true>(currObj, query_data,
                                                                                                           ef_);
                top_candidates.swap(top_candidates1);
            }
            else{
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<false>(currObj, query_data,
                                                                                                            ef_);
                top_candidates.swap(top_candidates1);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
			*/
            return top_candidates;
        };

        void resizeIndex(size_t new_max_elements){
            if (new_max_elements<cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");


            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);


            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }

        void saveIndex(const std::string &location, float diff3) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            //min_norm2_ = tol_min_norm;
			//diff2_ = diff2;
			diff3_ = diff3;

            writeBinaryPOD(output, m_);
			writeBinaryPOD(output, min_norm_);
			writeBinaryPOD(output, diff_);

			writeBinaryPOD(output, diffadj_);
			writeBinaryPOD(output, diffres_);
			writeBinaryPOD(output, difflast_);

			writeBinaryPOD(output, level_);
			writeBinaryPOD(output, vecdim0_);
			
            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

			writeBinaryPOD(output, size_data_per_element16_);
			writeBinaryPOD(output, size_data_per_element32_);
			writeBinaryPOD(output, size_data_per_element48_);
			
			writeBinaryPOD(output, size_links_level0_);
			writeBinaryPOD(output, size_links_level16_);
			writeBinaryPOD(output, size_links_level32_);
	        writeBinaryPOD(output, size_links_level48_);
	
            writeBinaryPOD(output, sec_part_);
	
			writeBinaryPOD(output, num16);
			writeBinaryPOD(output, num32);
			writeBinaryPOD(output, num48);
			writeBinaryPOD(output, num64);

            size_t new_size = size_data_per_element16_ * num16 + size_data_per_element32_ * num32+ size_data_per_element48_ * num48 + size_data_per_element_ * num64;
			output.write(data_level0_memory_, new_size);

            if(inverse_id == NULL) {printf("incorrectness\n"); exit(0);}

            for (size_t i = 0; i < cur_element_count; i++) {
				int pos = inverse_id[i];
                unsigned int linkListSize = element_levels_[pos] > 0 ? size_links_per_element_ * element_levels_[pos] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[pos], linkListSize);
            }
			
			delete[] inverse_id;
			
			size_t index_size = size_links_level16_ * num16 + size_links_level32_ * num32 + size_links_level48_ * num48 + size_links_level0_ * num64;	
			
		    data_level0_memory32_ =  data_level0_memory_ + size_links_level16_ * num16;
			data_level0_memory48_ =  data_level0_memory32_ + size_links_level32_ * num32;
          	data_level0_memory64_ =  data_level0_memory48_ + size_links_level48_ * num48;	   
		    vec_level0_memory_ = data_level0_memory_ + index_size;	
		
			num32 += num16;
            num48 += num32;			
			
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, SpaceInterface<dist_t> *s1, SpaceInterface<dist_t> *s2, SpaceInterface<dist_t> *s3, float diff3, int LSH_level, int LSH_vecdim0, size_t max_elements_i=0) {
            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0,input.end);
            std::streampos total_filesize=input.tellg();
            input.seekg(0,input.beg);

            readBinaryPOD(input, m_);
			readBinaryPOD(input, min_norm_);
			readBinaryPOD(input, diff_);
						
			readBinaryPOD(input, diffadj_);
			readBinaryPOD(input, diffres_);
            readBinaryPOD(input, difflast_);			
						
			readBinaryPOD(input, level_);
			readBinaryPOD(input, vecdim0_);
			
            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if(max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);
			
			readBinaryPOD(input, size_data_per_element16_);
			readBinaryPOD(input, size_data_per_element32_);
			readBinaryPOD(input, size_data_per_element48_);
			
			readBinaryPOD(input, size_links_level0_);
			readBinaryPOD(input, size_links_level16_);
			readBinaryPOD(input, size_links_level32_);
			readBinaryPOD(input, size_links_level48_);
			
			readBinaryPOD(input, sec_part_);
			
			readBinaryPOD(input, num16);
			readBinaryPOD(input, num32);
			readBinaryPOD(input, num48);
			readBinaryPOD(input, num64);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
			fstipfunc_ = s2->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

			fstPQfunc_ = s1->get_dist_func();
			fstLSHfunc_ = s3->get_dist_func();			
			subip_func_param_ = s1->get_dist_func_param();
			LSHip_func_param_ = s3->get_dist_func_param();


            LSH_level_ = LSH_level;
            LSH_vecdim0_ = LSH_vecdim0;			

			diff3_ = diff3;

            size_t new_size = size_data_per_element16_ * num16 + size_data_per_element32_ * num32+ size_data_per_element48_ * num48 + size_data_per_element_ * num64;

            auto pos=input.tellg();

            /// Optional - check if index is ok:

            input.seekg(new_size,input.cur);
            for (size_t i = 0; i < cur_element_count; i++) {
                if(input.tellg() < 0 || input.tellg()>=total_filesize){
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0) {
                    input.seekg(linkListSize,input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if(input.tellg()!=total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();

            /// Optional check end

            input.seekg(pos,input.beg);
			
			size_t index_size = size_links_level16_ * num16 + size_links_level32_ * num32 + size_links_level48_ * num48 + size_links_level0_ * num64;
			
			data_level0_memory_ = (char *) malloc(new_size);
		   if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
           input.read(data_level0_memory_, new_size);
		   
		   //-------------------------------------
		    data_level0_memory32_ =  data_level0_memory_ + size_links_level16_ * num16;
			data_level0_memory48_ =  data_level0_memory32_ + size_links_level32_ * num32;
          	data_level0_memory64_ =  data_level0_memory48_ + size_links_level48_ * num48;	   
		    vec_level0_memory_ = data_level0_memory_ + index_size;
			
			num32 += num16;
			num48 += num32;
		   //---------------------------------

            size_links_per_element_ = maxM_ * (4 * sizeof(unsigned short int) + sizeof(tableint) + LSH_level) + sizeof(linklistsizeint);

            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(max_update_element_locks).swap(link_list_update_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
			
            for (size_t i = 0; i < cur_element_count; i++) {
                //label_lookup_[getExternalLabel(i)]=i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    element_levels_[i] = 0;

                    linkLists_[i] = nullptr;
                } else {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            input.close();

            return;
        }

        template<typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label) const
        {
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;

            char* data_ptrv = getDataByInternalId(label_c);
            size_t dim = *((size_t *) dist_func_param_);
            std::vector<data_t> data;
            data_t* data_ptr = (data_t*) data_ptrv;
            for (int i = 0; i < dim; i++) {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        static const unsigned char DELETE_MARK = 0x01;
        // static const unsigned char REUSE_MARK = 0x10;
        /**
         * Marks an element with the given label deleted, does NOT really change the current graph.
         * @param label
         */
        void markDelete(labeltype label)
        {
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            markDeletedInternal(internalId);
        }

        /**
         * Uses the first 8 bits of the memory for the linked list to store the mark,
         * whereas maxM0_ has to be limited to the lower 24 bits, however, still large enough in almost all cases.
         * @param internalId
         */
        void markDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (!isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
                *ll_cur |= DELETE_MARK;
                num_deleted_ += 1;
            }
            else
            {
                throw std::runtime_error("The requested to delete element is already deleted");
            }
        }

        /**
         * Remove the deleted mark of the node, does NOT really change the current graph.
         * @param label
         */
        void unmarkDelete(labeltype label)
        {
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            unmarkDeletedInternal(internalId);
        }

        /**
         * Remove the deleted mark of the node.
         * @param internalId
         */
        void unmarkDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
                *ll_cur &= ~DELETE_MARK;
                num_deleted_ -= 1;
            }
            else
            {
                throw std::runtime_error("The requested to undelete element is not deleted");
            }
        }

        /**
         * Checks the first 8 bits of the memory to see if the element is marked deleted.
         * @param internalId
         * @return
         */
        bool isMarkedDeleted(tableint internalId) const {
            unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId))+2;
            return *ll_cur & DELETE_MARK;
        }

        unsigned short int getListCount(linklistsizeint * ptr) const {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint * ptr, unsigned short int size) const {
            *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
        }

        void addPoint(const void *data_point, labeltype label, float* norm) {
            addPoint(data_point, label, -1, norm);
        }
	
        void CalcEdgeNorm(int id, int vecdim_, double* edge_norm, size_t* count){
			
            int* data;
			float* diff_data = new float[vecdim_];
						   
			//for(int id = 0; id < data_size; id++){
                data = (int *) get_linklist0(id);
                size_t size = getListCount((linklistsizeint*)data);
			
                int* datal = data + 1;

                float* center_data = (float*) getDataByInternalId(id);
                
				(*count) += size;
				
				for (int i = 0; i < size; i++) {
				    int a = *getNeighborid(datal, i);

                    float* obj_data = (float*) getDataByInternalId(a);

                    calc_vec(obj_data, center_data, diff_data, vecdim_);

                    for(int l = 0; l < vecdim_; l++){
					    edge_norm[l] += diff_data[l] * diff_data[l];
				    }			
			    }
			//}
			delete[] diff_data;
		}	
		
        void CalcEdgeVariance(double* norm2, int vecdim_, int data_size, double* avg){
			
            int* data;
			float* diff_data = new float[vecdim_];
						   
			for(int id = 0; id < data_size; id++){
                data = (int *) get_linklist0(id);
                size_t size = getListCount((linklistsizeint*)data);
			
                int* datal = data + 1;

                float* center_data = (float*) getDataByInternalId(id);
                
				for (int i = 0; i < size; i++) {
				    int a = *getNeighborid(datal, i);

                    float* obj_data = (float*) getDataByInternalId(a);

                    for(int l = 0; l < vecdim_; l++){
					    diff_data[l] = obj_data[l] - center_data[l] - avg[l];
					    norm2[l] += (diff_data[l] * diff_data[l]);
				    }			
			    }
			}
			delete[] diff_data;
		}		
	
        void PermuteVec(int id, float** vec, int vecdim_){
            float* center_data = (float*) getDataByInternalId(id);
			int y = *getExternalLabeLp(id);
			for(int i = 0; i < vecdim_; i++)
				center_data[i] = vec[y][i];			
		}

        void Calc_wres(size_t edge_count, int vecdim0, int vecdim_, int data_size, int level){
			
            int* data;
			float* diff_data = new float[vecdim_];
			
            double sqrt_L = sqrt(1.0 / level);
            double tol_sum = 0;	

            size_t real_count = 0;			
			for(int id = 0; id < data_size; id++){
                data = (int *) get_linklist0(id);
                size_t size = getListCount((linklistsizeint*)data);
			
                int* datal = data + 1;

                float* center_data = (float*) getDataByInternalId(id);
                
				for (int i = 0; i < size; i++) {
				    int a = *getNeighborid(datal, i);

                    float* obj_data = (float*) getDataByInternalId(a);

                    for(int l = 0; l < vecdim_; l++){
					    diff_data[l] = obj_data[l] - center_data[l];
				    }
					
					double sum2 = 0;
                    for(int l = 0; l < vecdim_; l++){
					    sum2 += diff_data[l] * diff_data[l];
				    }

                    if (sum2 > 0.00000001){
						real_count++;
						sum2 = sqrt(sum2);

                        for(int l = 0; l < vecdim_; l++){
					        diff_data[l] = diff_data[l] / sum2;
				        }							

                        float temp_sum = 0;
                        for(int l = 0; l < level; l++){						
						
						    float* cur_data = diff_data + l * vecdim0;
						    float sum = 0;
                            for(int ll = 0; ll < vecdim0; ll++){
							    sum += cur_data[ll] * cur_data[ll];
						    }
						    temp_sum += sqrt(sum) * sqrt_L;
				        }
						tol_sum += temp_sum;
				   	}					
			    }
			}
			double avg_res = tol_sum / real_count;
		//	printf("avg_res = %lf; real_count = %ld\n", avg_res, real_count);
			delete[] diff_data;
		}
		

        float calc_vec(float* obj_vec, float* cen_vec, float* diff_vec, int vecdim){
			
			float* tmp_cen = new float[vecdim];
			for(int i = 0; i < vecdim; i++)
				tmp_cen[i] = cen_vec[i]; 
			
			double obj_norm = fstipfunc_(obj_vec, obj_vec, dist_func_param_);
		    obj_norm = sqrt(obj_norm);
			
			double cen_norm = fstipfunc_(tmp_cen, tmp_cen, dist_func_param_);
			cen_norm = sqrt(cen_norm);
			
			float scale = 0;
			float min_real = 0.000001;
			if(cen_norm < min_real || obj_norm < min_real){
	            for(int i = 0; i < vecdim; i++){
				    tmp_cen[i] = 0;
			    }				
			}
		    else{
			    float cos = (float) (fstipfunc_(obj_vec, tmp_cen, dist_func_param_) / obj_norm / cen_norm);
			    scale = (float) (obj_norm * cos / cen_norm);
				
	            for(int i = 0; i < vecdim; i++){
				    tmp_cen[i] = tmp_cen[i] * scale;
			    }
                scale = obj_norm * cos;				
		    }
	        for(int i = 0; i < vecdim; i++){
				diff_vec[i] = obj_vec[i] - tmp_cen[i];
			}		
			
			delete[] tmp_cen;
			return scale;		
		}
		
        float calc_vec_adjust(float* obj_vec, float* cen_vec, float* diff_vec, int vecdim, bool* is_zero, bool* is_edge){
			
			float* tmp_cen = new float[vecdim];
			for(int i = 0; i < vecdim; i++)
				tmp_cen[i] = cen_vec[i]; 
			
			double obj_norm = fstipfunc_(obj_vec, obj_vec, dist_func_param_);
		    obj_norm = sqrt(obj_norm);
			
			double cen_norm = fstipfunc_(tmp_cen, tmp_cen, dist_func_param_);
			cen_norm = sqrt(cen_norm);
			
			float scale = 0;
			float min_real = 0.000001;
			if(cen_norm < min_real || obj_norm < min_real){
	            for(int i = 0; i < vecdim; i++){
				    tmp_cen[i] = 0;
			    }
                *is_edge = true;
                if(cen_norm < min_real){
			    *is_zero = true;		
				}
				
			}
		    else{
			    float cos = (float) (fstipfunc_(obj_vec, tmp_cen, dist_func_param_) / obj_norm / cen_norm);
			    scale = (float) (obj_norm * cos / cen_norm);
				
                if(scale > 1000000000){
					printf("overlarge scale\n");
					exit(0);
				}				
				
	            for(int i = 0; i < vecdim; i++){
				    tmp_cen[i] = tmp_cen[i] * scale;
			    }
				scale = obj_norm * cos;	
				
		    }
	        for(int i = 0; i < vecdim; i++){
				diff_vec[i] = obj_vec[i] - tmp_cen[i];
			}		
			
			delete[] tmp_cen;
			return scale;		
		}	

        float calc_res(float* LSH_data, float* cen_vec, bool flag){
			double cen_norm = fstipfunc_(cen_vec, cen_vec, dist_func_param_);
			cen_norm = sqrt(cen_norm);

			if(flag == true){
                return 0;				
			}
		    else{
			    float ip = (float) (fstipfunc_(cen_vec, LSH_data, dist_func_param_) / cen_norm);
		        return ip;
		    }	
		}		
			
	
void addProjVal(int id, float*** LSH_vec, float* tmp_norm2, float* tmp_adjust, float* tmp_res,  float* tmp_last, int vecdim_, float* max_norm2, float* max_adjust, float* max_res, float* max_last, unsigned short int* norm_quan, float* val, bool* is_zero, bool* is_edge){
	        //printf("check1\n");
            bool cur_sign, max_sign;
			int cur_ip;
		    float max_sum;
		    unsigned char max_ip;
			
            float* diff_data = new float[vecdim_];
			//float* LSH_data = new float[vecdim_];
			float* LSH_data;
            float* cen_vec = new float[vecdim_];
			//unsigned char* proj_info = new unsigned char[level_];
            unsigned char* LSH_info = new unsigned char[LSH_level_];			

            int* data;
			
            data = (int *) get_linklist0(id);
			
            //int bb = getExternalLabel(id);
            size_t size = getListCount((linklistsizeint*)data);
			
            int* datal = data + 1;

            float* center_data = (float*) getDataByInternalId(id);			

//printf("check0.1\n");	
            for (int i = 0; i < size; i++) {
				int a = *getNeighborid(datal, i);
                float* obj_data = (float*) getDataByInternalId(a);

                float test_norm = 0;

                for(int l = 0; l < vecdim_; l++){
					diff_data[l] = obj_data[l] - center_data[l];
					test_norm += (diff_data[l] * diff_data[l]);
				}
				
				test_norm = sqrt(test_norm);

                is_edge[i] = false; 
				tmp_adjust[i] = calc_vec_adjust(obj_data, center_data, diff_data, vecdim_, is_zero, &(is_edge[i]));
	
                float vval = tmp_adjust[i];
				if(vval < 0) vval = -1 * tmp_adjust[i];

				if(i == 0) {
					*max_adjust = vval;
				}
				else{
				    if(vval > *max_adjust) {*max_adjust = vval;}
  				}
				
                LSH_data = diff_data;
				//----------LSH nearest vector--------
				
			    for(int k = 0; k < LSH_level_; k++){
			
		            for(int j = 0; j < m_; j++){
			            float sum = 0;
			            for(int l = 0; l < LSH_vecdim0_; l++){
			                sum += LSH_data[k * LSH_vecdim0_ + l] * LSH_vec[k][j][l];
			            }
	
				        if(sum < 0) {sum = -1.0f * sum; cur_ip = j + 128;}
				        else{cur_ip = j;}
				
				        if(j == 0) {max_sum = sum; max_ip = cur_ip;}
				        else{
					        if(sum > max_sum) {max_sum = sum; max_ip = cur_ip;}
				        }
		            }
				    LSH_info[k] = max_ip;						
			    }
                //------------compute last norm---------------------------------
				//printf("check0.2, vecdim_ = %d\n", vecdim_);	
				for(int k = 0; k < LSH_level_; k++){
					//printf("LSH_level = %d, LSH_vecdim0_ = %d, k = %d\n", LSH_level_, LSH_vecdim0_ , k);
					//int ttttt = LSH_info[k];
					//printf("id = %d\n", ttttt);
			        for(int l = 0; l < LSH_vecdim0_; l++){
						//printf("l = %d,  val1 = %d, k = %d, LSH_info[k] = %d\n",l, k * LSH_vecdim0_ + l, k, LSH_info[k]);
						int s;
						float s1;
						if(LSH_info[k] >= 128){
							s = LSH_info[k] - 128;
							s1 = -1.0f;
						}
						else{
							s = LSH_info[k];
							s1 = 1.0f;
						}
						
			            cen_vec[k * LSH_vecdim0_ + l] =  LSH_vec[k][s][l] * s1;
			        }					
				}
				
				tmp_last[i] = calc_res(cen_vec, center_data, *is_zero);				
						
                vval = tmp_last[i];
				if(vval < 0) vval = -1 * tmp_last[i];

				if(i == 0) {
					*max_last = vval;
				}
				else{
				    if(vval > *max_last) {*max_last = vval;}
  				}



				//------------------------------------------------------------------------

                tmp_norm2[i] = 0;

                for(int l = 0; l < vecdim_; l++){
					tmp_norm2[i] += (LSH_data[l] * LSH_data[l]);
				}

				tmp_norm2[i] = sqrt(tmp_norm2[i]);
				
                //--------test--------------------------
				
				if(test_norm > 0.000001){
					//*val += test_norm2/ test_norm;
				    *val += tmp_norm2[i] / test_norm;
				}
				//--------test end-------------------------
				
				if(i == 0) {*max_norm2 = tmp_norm2[i];}
				else{
				    if(tmp_norm2[i] > *max_norm2) {*max_norm2 = tmp_norm2[i];}
  				}				
				
				//-------------------------------
				
				tmp_res[i] = calc_res(LSH_data, center_data, *is_zero);
				if(i == 0) {*max_res = tmp_res[i];}
				else{
				    if(tmp_res[i] > *max_res) {*max_res = tmp_res[i];}
  				}					
				
                tableint cand = getExternalLabel(a);
                if (cand < 0 || cand > max_elements_)
                   throw std::runtime_error("cand error");
	            
				//printf("check0.4\n");
				
			    if(size > 48){
				    unsigned short int* norm_pointer = getFirstNorm(datal, i);
				    *norm_pointer = norm_quan[cand]; 
				}
				else if(size > 32){
				    unsigned short int* norm_pointer = getFirstNorm48(datal, i);
				    *norm_pointer = norm_quan[cand]; 					
				}
				else if(size > 16){
				    unsigned short int* norm_pointer = getFirstNorm32(datal, i);
				    *norm_pointer = norm_quan[cand]; 					
				}
				else{
				    unsigned short int* norm_pointer = getFirstNorm16(datal, i);
				    *norm_pointer = norm_quan[cand]; 					
				}
								
				if(size > 48){
									for(int j = 0; j < LSH_level_; j++){
					unsigned char* pointer = getLSHM(datal, j, i);
					*pointer = LSH_info[j];
									}
				}
				else if(size > 32){
									for(int j = 0; j < LSH_level_; j++){
					unsigned char* pointer = getLSHM48(datal, j, i);
					*pointer = LSH_info[j];					
									}
				}
				else if(size > 16){
									for(int j = 0; j < LSH_level_; j++){
					unsigned char* pointer = getLSHM32(datal, j, i);
					*pointer = LSH_info[j];				
									}
				}
                else{
				    for(int j = 0; j < LSH_level_; j++){
					unsigned char* pointer = getLSHM16(datal, j, i);
					*pointer = LSH_info[j];				
					}	
                }				
			}
			
			//delete[] proj_info;
            delete[] LSH_info;	
			delete[] diff_data;
			//delete[] LSH_data;
			delete[] cen_vec;
		}



        void addEdgeNorm(int id, float* tmp_norm2, float* tmp_adjust, float* tmp_res, float* tmp_last, float diff2, float diffadj, float diffres, float difflast, bool* flag){
			
            int* data;
			
            data = (int *) get_linklist0(id);
            size_t size = getListCount((linklistsizeint*)data);
			
            int* datal = data + 1;
			
            for (int i = 0; i < size; i++) {
				int a = *getNeighborid(datal, i);
				
                tableint cand = getExternalLabel(a);
                if (cand < 0 || cand > max_elements_)
                   throw std::runtime_error("cand error");
			
			    unsigned short int b;
			    int c;
			
                //-----------write adjust----------------
				short int b0;				
				c = (tmp_adjust[i]) / diffadj;
                
				if(c > 32767) {
				    b0 = 32767;
			    }
                else if(c < -32767) {
				    b0 = -32767;
			    }				
                else{
				    b0 = c;
			    }

				if(flag[i] == true) b0 = 0;				
					
				if(size > 48)
				    *getAdjustNorm(datal, i) = b0;	
				else if(size > 32)
				    *getAdjustNorm48(datal, i) = b0;	
				else if(size > 16)
				    *getAdjustNorm32(datal, i) = b0;
                else 
                    *getAdjustNorm16(datal, i) = b0;
				//---------------------------------------
				//---------------------------------------
			    c = (tmp_norm2[i]) / diff2;
				c++;
		        if(c < 1) {b = 1;}  //!!LSH norm cannot be 0
                else if(c > 65535) {
				    b = 65535;
			    }
                else{
				    b = c;
			    }
					
				//if(flag == true) b = 0;	
					
				if(size > 48)
				    *getLSHNorm(datal, i) = b;	
				else if(size > 32)
				    *getLSHNorm48(datal, i) = b;	
				else if(size > 16)
				    *getLSHNorm32(datal, i) = b;
				else
					*getLSHNorm16(datal, i) = b;
                //---------------------------------------------------
			    c = (tmp_last[i]) / difflast;
				
				if(c > 32767) {
				    b0 = 32767;
			    }
                else if(c < -32767) {
				    b0 = -32767;
			    }				
                else{
				    b0 = c;
			    }

				if(flag[i] == true) b0 = 0;
				
				if(size > 48)
				    *getLastNorm(datal, i) = b0;	
				else if(size > 32)
				    *getLastNorm48(datal, i) = b0;	
				else if(size > 16)
				    *getLastNorm32(datal, i) = b0;
                else
                    *getLastNorm16(datal, i) = b0;						
			}
			
			diffadj_ = diffadj;
			diffres_ = diffres;
			difflast_ = difflast;
		}

		
		void find_neighbors(size_t vecdim_, float** train_org, int ind, int* count, float** data){ //check
		
		    float min_real = 0.000001;
			int true_id = getExternalLabel(ind);
			
            int* datal = (int *) get_linklist0(ind);
            size_t size = getListCount((linklistsizeint*)datal);
			
            datal = datal + 1;			
			
			for(int i = 0; i < size; i++){
				//int obj_id = getExternalLabel(neighbors[i]);
				int obj_id = *getNeighborid(datal, i);
				tableint cand = getExternalLabel(obj_id);
				float sum = 0;
				
				float* tmp = new float[vecdim_];
				calc_vec(data[cand], data[true_id], tmp, vecdim_);
				
				for(int j = 0; j < vecdim_; j++){
					//train_org[*count][j] = data[cand][j] - data[true_id][j];
					train_org[*count][j] = tmp[j];
                    sum += train_org[*count][j] * train_org[*count][j];					
				}
				
				delete[] tmp;
				
				if (sum < min_real) continue; 
				
				float ssum = sqrt(sum);
				
				for(int j = 0; j < vecdim_; j++){
					train_org[*count][j] = train_org[*count][j] / ssum;				
				}				
				
				(*count)++;
				if( *count >= size_n ) break;
			}
		}
		
		void graph_rot(int cur_id, int vecdim_, float* R){
			float* data = (float *) getDataByInternalId(cur_id);
			float* data3 = new float[vecdim_]; 
		    for(int j = 0; j < vecdim_; j++){
				data3[j] = fstipfunc_( (const void*) (R + j * vecdim_), (const void*) (data), dist_func_param_);		
		    }			
			memcpy(data, data3, sizeof(float) * vecdim_);
			delete[] data3;
		}		
		

        void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++) {
                std::unordered_set<tableint> sCand;
                std::unordered_set<tableint> sNeigh;
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto&& elOneHop : listOneHop) {
                    sCand.insert(elOneHop);

                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto&& elTwoHop : listTwoHop) {
                        sCand.insert(elTwoHop);
                    }
                }

                for (auto&& neigh : sNeigh) {
                    // if (neigh == internalId)
                    //     continue;

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1; // sCand guaranteed to have size >= 1
                    size_t elementsToKeep = std::min(ef_construction_, size);
                    for (auto&& cand : sCand) {
                        if (cand == neigh)
                            continue;

                        dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                        if (candidates.size() < elementsToKeep) {
                            candidates.emplace(distance, cand);
                        } else {
                            if (distance < candidates.top().first) {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // Retrieve neighbours using heuristic and set connections.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    {
                        std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        size_t candSize = candidates.size();
                        setListCount(ll_cur, candSize);
                        tableint *data = (tableint *) (ll_cur + 1);
                        for (size_t idx = 0; idx < candSize; idx++) {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        };

        void repairConnectionsForUpdate(const void *dataPoint, tableint entryPointInternalId, tableint dataPointInternalId, int dataPointLevel, int maxLevel) {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel) {
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj,level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            for (int level = dataPointLevel; level >= 0; level--) {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                        currObj, dataPoint, level);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0) {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0) {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted) {
                        filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
                }
            }
        }

        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
            std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *) (data + 1);
            memcpy(result.data(), ll,size * sizeof(tableint));
            return result;
        };

        tableint addPoint(const void *data_point, labeltype label, int level, float* norm) {
            //printf("check internal 1\n");
            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) {
                    tableint existingInternalId = search->second;
                    templock_curr.unlock();

                    std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);

                    if (isMarkedDeleted(existingInternalId)) {
                        unmarkDeletedInternal(existingInternalId);
                    }
                    updatePoint(data_point, existingInternalId, 1.0);
                    exit(0);
                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
			
			//printf("check internal 2\n");
            if (level > 0)
                curlevel = level;

            element_levels_[cur_c] = curlevel;


            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;


            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
			
			memcpy(getNormByInternalId(cur_c), norm, sizeof(float));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);

            if (curlevel) {
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed)currObj != -1) {

                if (curlevel < maxlevelcopy) {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--) {


                        bool changed = true;
                        while (changed) {
                            changed = false;
                            int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                            data = (int *)get_linklist(currObj,level);
                            int size = getListCount((linklistsizeint*)data);

                            int* datal = data + 1;
                            for (int i = 0; i < size; i++) {
								int* data2 = getNeighborid(datal, i);
                                int cand = *data2;
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }
                
                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
	
                    if (epDeleted) {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                }

            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;

            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };


		void query_rot(int dim_, int qsize, float* massQ, float* R){
            float* data3 = new float[dim_];

            for(int i = 0; i < qsize; i++){		
	            for(int j = 0; j < dim_; j++){
		            data3[j] = fstipfunc_( (const void*) (R + j * dim_), (const void*) (massQ + i * dim_), dist_func_param_);				        
	            }
                memcpy(massQ + i * dim_, data3, sizeof(float) * dim_);		
            }
			delete[] data3;
		}

        void searchKnn(float *query_data, size_t k, unsigned int* result, float*** lsh_vec, float* thres_pos, float** query_lsh, int table_size) const {
            //std::priority_queue<std::pair<dist_t, labeltype >> result;
            //if (cur_element_count == 0) return result;			

            tableint currObj = enterpoint_node_;

		char* norm0 = getNormByInternalIdQuery(enterpoint_node_);
			float true_norm0 = *((float*) norm0); 
			norm0 += 4;
			dist_t curdist = fstipfunc_(query_data, norm0, dist_func_param_);
			curdist = true_norm0 * ( true_norm0 * 0.5 - curdist);

            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;

                    int *datal = (int *) (data + 1);
                    for (int i = 0; i < size; i++) {
						int* data2 = getNeighborid(datal, i);
                        int cand = *data2;
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
						
						char* norm = getNormByInternalIdQuery(cand);

			            float true_norm = *((float*) norm); 
		            	norm += 4;
			            dist_t d = fstipfunc_(query_data, norm, dist_func_param_);
			            d = true_norm * ( true_norm * 0.5 - d);

                        if (d < curdist) {
                            curdist = d;
							//printf("cur_dist = %f\n", curdist);
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
			
            if (num_deleted_) {
                searchBaseLayerST<true,true>(
                        currObj, query_data, std::max(ef_, k), result, k, lsh_vec, thres_pos, query_lsh, table_size);
            }
            else{
                searchBaseLayerST<false,true>(
                        currObj, query_data, std::max(ef_, k), result, k, lsh_vec, thres_pos, query_lsh, table_size);
            }
			//printf("check2.2\n");
        };


        void checkIntegrity(){
            int connections_checked=0;
            std::vector <int > inbound_connections_num(cur_element_count,0);
            for(int i = 0;i < cur_element_count; i++){
                for(int l = 0;l <= element_levels_[i]; l++){
                    linklistsizeint *ll_cur = get_linklist_at_level(i,l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *) (ll_cur + 1);
                    std::unordered_set<tableint> s;
                    for (int j=0; j<size; j++){
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert (data[j] != i);
                        inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;

                    }
                    assert(s.size() == size);
                }
            }
            if(cur_element_count > 1){
                int min1=inbound_connections_num[0], max1=inbound_connections_num[0];
                for(int i=0; i < cur_element_count; i++){
                    assert(inbound_connections_num[i] > 0);
                    min1=std::min(inbound_connections_num[i],min1);
                    max1=std::max(inbound_connections_num[i],max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";

        }
		
        void compression(int vecsize, int vecdim, bool* is_zero){
			num16 = 0;
			num32 = 0;
			num48 = 0;
			num64 = 0;
			
			int* convert_id = new int[vecsize];
			int* convert_id2 = new int[vecsize];
			inverse_id = new int[vecsize];
			
			char* indicator= new char[vecsize];
			
            for(int i = 0; i < vecsize; i++){
				int* data = (int *)get_linklist0(i);
				size_t count = getListCount((linklistsizeint*)data);
				data += 1;
				
				if(count > 48){
					indicator[i] = 3;
					convert_id[i] = num64;
					num64++;
				}
				else if(count > 32){
					indicator[i] = 2;
					convert_id[i] = num48;
					num48++;					
				}
				else if(count > 16){
					indicator[i] = 1;
					convert_id[i] = num32;
					num32++;
				}
				else{
					indicator[i] = 0;
					convert_id[i] = num16;
					num16++;
				}
			}
			
			
           for(int i = 0; i < vecsize; i++){           //normalize data
				double res = fstipfunc_(getDataByInternalId(i), getDataByInternalId(i), dist_func_param_);
				res = sqrt(res);
 				*(float *)getNormByInternalId(i) = (float)res;				
                if(is_zero[i] == true)
					continue;
				else{
				    float* data = (float *) getDataByInternalId(i);
				    for(int j = 0; j < vecdim; j++){
					     data[j] = (float) (data[j] / res);
				    }
				}

			}			
			
			
   
            sec_part_ = data_size_ + sizeof(labeltype) + sizeof(float);  
            size_links_level16_ = 16 * (4 * sizeof(unsigned short int) + sizeof(tableint) +LSH_level_ ) + sizeof(linklistsizeint);   //new
            size_data_per_element16_ = size_links_level16_ + data_size_ + sizeof(labeltype) + sizeof(float); 
			
            size_links_level32_ = 32 * (4 * sizeof(unsigned short int) + sizeof(tableint) + LSH_level_) + sizeof(linklistsizeint);   //new
            size_data_per_element32_ = size_links_level32_ + data_size_ + sizeof(labeltype) + sizeof(float); 

            size_links_level48_ = 48 * (4 * sizeof(unsigned short int) + sizeof(tableint) + LSH_level_) + sizeof(linklistsizeint);   //new
            size_data_per_element48_ = size_links_level48_ + data_size_ + sizeof(labeltype) + sizeof(float);  
 

			size_t new_size = size_data_per_element16_ * num16 + size_data_per_element32_ * num32+ size_data_per_element48_ * num48 + size_data_per_element_ * num64;
				
		    size_t index_size = size_links_level16_ * num16 + size_links_level32_ * num32 + size_links_level48_ * num48 + size_links_level0_ * num64;
				
            char* data_level0_memory_new_ = (char *) malloc(new_size);
		        if (data_level0_memory_new_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
   
            vec_level0_memory_ = data_level0_memory_new_ + index_size;
   
            int a = (int)num16;
            int b = (int)(num16 + num32);
			int c = (int)(num16 + num32 + num48);

                for(int i = 0; i < vecsize; i++){
					if(indicator[i] == 0){
						convert_id2[i] = convert_id[i];
					}
					else if(indicator[i] == 1){
						convert_id2[i] = convert_id[i] + a;
					}
					else if(indicator[i] == 2){
						convert_id2[i] = convert_id[i] + b;
					}					
					else{
						convert_id2[i] = convert_id[i] + c;
					}
				}
				
            for(int i = 0; i < vecsize; i++){
				int* data = (int *)get_linklist0(i);
				size_t count = getListCount((linklistsizeint*)data);
				data += 1;
				
				for(int i = 0; i < count; i++){
					data[i] = convert_id2[data[i]];
				}
			}				

                char* pos_16 = data_level0_memory_new_;
                char* pos_32 = data_level0_memory_new_ + size_links_level16_ * num16;
                char* pos_48 = data_level0_memory_new_ + size_links_level16_ * num16 + size_links_level32_ * num32;				
				char* pos_64 = data_level0_memory_new_ + size_links_level16_ * num16 + size_links_level32_ * num32 + size_links_level48_ * num48;

                for(int i = 0; i < vecsize; i++){
					if(indicator[i] == 0){
						int pos = convert_id[i];
						char* cen = data_level0_memory_  + (size_data_per_element_ *  i);
						char* obj = pos_16 + (size_links_level16_ * pos);
						memcpy(obj, cen, size_links_level16_);
						
						cen += size_links_level0_;
						obj = vec_level0_memory_ + (pos * sec_part_);
						
						memcpy(obj, cen, sec_part_);
						
					}
					else if(indicator[i] == 1){
						int pos = convert_id[i];
						char* cen = data_level0_memory_  + (size_data_per_element_ *  i);
						char* obj = pos_32 + (size_links_level32_ * pos);
						memcpy(obj, cen, size_links_level32_);
						
						pos += a;
						cen += size_links_level0_;
						obj = vec_level0_memory_ + (pos * sec_part_);
						//obj += size_links_level32_;
						
						memcpy(obj, cen, sec_part_);
					}
					else if(indicator[i] == 2){
						int pos = convert_id[i];
						char* cen = data_level0_memory_  + (size_data_per_element_ *  i);
						char* obj = pos_48 + (size_links_level48_ * pos);
						memcpy(obj, cen, size_links_level48_);
						
						pos += b;
						cen += size_links_level0_;
						obj = vec_level0_memory_ + (pos * sec_part_);					
						memcpy(obj, cen, sec_part_);
					}										
					else{
						int pos = convert_id[i];
						char* cen = data_level0_memory_  + (size_data_per_element_ *  i);
						char* obj = pos_64 + (size_links_level0_ * pos);
						memcpy(obj, cen, size_links_level0_);
						
						pos += c;
						cen += size_links_level0_;
						obj = vec_level0_memory_ + (pos * sec_part_);					
						memcpy(obj, cen, sec_part_);
					}
				}

            for (int i = 0; i < vecsize; i++) {
                if(element_levels_[i] <= 0) continue;
                
                int cur_level = element_levels_[i];				
				unsigned int *data; 
				 
                for(int j = 1; j <= cur_level; j++){
                    data = (unsigned int *) get_linklist(i, j);
                    size_t size = getListCount(data);
                    int *datal = (int *) (data + 1);
                    for (int l = 0; l < size; l++) {
						int* data2 = getNeighborid(datal, l);
                        int cand = *data2;
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
						
						*data2 = convert_id2[cand];
                    }
                }
            }

            for(int i = 0; i < vecsize; i++){
                inverse_id[convert_id2[i]] = i;  			
			}		
						
			enterpoint_node_ = convert_id2[enterpoint_node_];
			delete[] convert_id;
			delete[] convert_id2;
            delete[] indicator;	
  
            if(cur_element_count != vecsize) {printf("incorrectness_inequality\n"); exit(0);}       
	   
            label_offset_ = data_size_ + sizeof(float);
            free(data_level0_memory_);
            data_level0_memory_ = data_level0_memory_new_;
            vec_level0_memory_ = data_level0_memory_ + index_size;			
				
			
        }

        void Boost(int* new_edge, int* new_size) {
			
			int eff = 10000;
            int *data_i = (int *) get_linklist0(enterpoint_node_);
            size_t pp = getListCount((linklistsizeint*)data_i);
			
			if(pp < maxM0_){
                BoostBaseLayer(enterpoint_node_, getDataByInternalId(enterpoint_node_), eff, new_edge, new_size);
			}
			return;
        };

        void Add_Edge(int* new_edge, int new_size) {
			
            int *data_i = (int *) get_linklist0(enterpoint_node_);
			size_t pp = getListCount((linklistsizeint*)data_i);
			int *datal = data_i + 1;
			
			if(pp < maxM0_){
			    for(int i = 0; i < new_size; i++){
				    datal[i] = new_edge[i]; 
			    }				
                setListCount((linklistsizeint*)data_i, (size_t) new_size);
			}			
			return;
        };

        void BoostBaseLayer(tableint ep_id, const void *data_point, size_t ef, int* new_edge, int *new_size) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

			int LL = ef;
            std::vector<Neighbor> retset(LL + 1);
            
            float w = 0.2;			
            float max_dist = 0;
			int max_ip = 0;
			bool cur_sign = 1;
			bool max_sign = 1;
			int step = 100;
						
			int vecdim = vecdim0_ * level_;	            		
		   			
			char* norm_pointer0 = getNormByInternalId(ep_id);
			float half_square_norm0 = *(float *) norm_pointer0;
			norm_pointer0 += 4;
			float ip0 = fstipfunc_(data_point, norm_pointer0, dist_func_param_);
            float dist0 = half_square_norm0 - ip0;
		
            retset[0] = Neighbor(ep_id, dist0, ip0, true);
            visited_array[ep_id] = visited_array_tag; 

            int k = 0;
			int l_num = 1;

	
            while (k < LL) {
                int nk = LL;

                if (retset[k].flag) {
                    retset[k].flag = false;
                    unsigned n = retset[k].id;
					
					if(l_num < LL){	
                        int *data = (int *) get_linklist0(n);
                        size_t size = getListCount((linklistsizeint*)data);
						int* datal = data + 1;
				        int* data2 = getNeighborid(datal, 0);
						int* data3 = getNeighborid(datal, 1);		

#ifdef USE_SSE
                        _mm_prefetch((char *) (visited_array + *data2), _MM_HINT_T0);
                        _mm_prefetch((char *) (visited_array + *data2 + 64), _MM_HINT_T0);
                        _mm_prefetch(data_level0_memory_ + (*data2) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                        _mm_prefetch((char *) (data3), _MM_HINT_T0);						
						
						
#endif

                        for (size_t j = 1; j <= size; j++) {					
                            //int candidate_id = *(data + j);
							int candidate_id = *(getNeighborid(datal, j-1));
							
							//printf("j = %d, id = %d, eid = %d\n", j, candidate_id, *getExternalLabeLp(candidate_id));
							int* data4 = getNeighborid(datal, j);
							             
#ifdef USE_SSE
                            _mm_prefetch((char *) (visited_array + *data4), _MM_HINT_T0);
                            _mm_prefetch(data_level0_memory_ + (*data4) * size_data_per_element_ + offsetData_,
                            _MM_HINT_T0);////////////

#endif
                            if (!(visited_array[candidate_id] == visited_array_tag)) {
                        
                                visited_array[candidate_id] = visited_array_tag;
	
			                    char* norm_pointer = getNormByInternalId(candidate_id);
			                    float half_square_norm = *(float *) norm_pointer;
			                    norm_pointer += 4;
                                float ip_ = fstipfunc_(data_point, norm_pointer, dist_func_param_);								
                                float dist = half_square_norm - ip_;
								
                                if (l_num == LL && dist >= retset[LL - 1].distance ) continue;

                                int r;
		                        if(l_num == LL){
 			
                 		            Neighbor nn2(candidate_id, dist, ip_, true);
                                    r = InsertIntoPool(retset.data(), LL, nn2);
		                        }
	                 	        else {
                                    Neighbor nn(candidate_id, dist, ip_, true);
                                    r = InsertIntoPool(retset.data(), l_num, nn);
			                        l_num++;
                                }
                                if (r < nk) {nk = r;}
						    }							
					    }
					}
					else{
						float val = retset[LL-1].distance; 
						float Lbound = retset[k].ip;
						
                        int *data = (int *) get_linklist0(n);
                        size_t size = getListCount((linklistsizeint*)data);

                        int* datal = data + 1;						

						int* data2 = getNeighborid(datal, 0);
						int* data3 = getNeighborid(datal, 1);

#ifdef USE_SSE
                        _mm_prefetch((char *) (visited_array + *data2), _MM_HINT_T0);
                        _mm_prefetch((char *) (visited_array + *data2 + 64), _MM_HINT_T0);
                        _mm_prefetch(data_level0_memory_ + (*data2) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                        _mm_prefetch((char *) (data3), _MM_HINT_T0);
#endif
                        for (size_t j = 0; j < size; j++) {
							int candidate_id = *getNeighborid(datal, j);
							int* data4 = getNeighborid(datal, j+1);

#ifdef USE_SSE
                            _mm_prefetch((char *) (visited_array + *data4), _MM_HINT_T0);
                            _mm_prefetch(data_level0_memory_ + (*data4) * size_data_per_element_ + offsetData_,
                            _MM_HINT_T0);////////////
#endif							       
                            if (!(visited_array[candidate_id] == visited_array_tag)) {
								visited_array[candidate_id] = visited_array_tag;
                                //float thres2 = norm_val - val;
								//if(is_checked[j] == false){continue;}
								
			                    char* norm_pointer = getNormByInternalId(candidate_id);
			                    float half_square_norm = *(float *) norm_pointer;
			                    norm_pointer += 4;
                                float ip_ = fstipfunc_(data_point, norm_pointer, dist_func_param_);								
                                float dist = half_square_norm - ip_;
								
                                if (l_num == LL && dist >= retset[LL - 1].distance ) continue;

                                int r;
		                        if(l_num == LL){
 			
                 		            Neighbor nn2(candidate_id, dist, ip_, true);
                                    r = InsertIntoPool(retset.data(), LL, nn2);
		                        }
	                 	        else {
                                    Neighbor nn(candidate_id, dist, ip_, true);
                                    r = InsertIntoPool(retset.data(), l_num, nn);
			                        l_num++;									
                                }
                                if (r < nk) {nk = r;}						
							}												
					    }				
					} 					
				}
				if (nk <= k)
                k = nk;
                else {++k;}
			}
			
            //---------selected neighbors(optional)------
			for(int i = 0; i < l_num; i++){
				retset[i].distance = fstdistfunc_(getDataByInternalId(ep_id), getDataByInternalId(retset[i].id), dist_func_param_);			
			}
			
            int *data_i = (int *) get_linklist0(ep_id);
            size_t size_i = getListCount((linklistsizeint*)data_i);
			int* datall = data_i + 1;
			int pp = size_i;
			
			printf("before boosting; size = %d\n", pp);
			
			for(int i = 0; i < pp; i++){
				new_edge[i] = datall[i];
			}
			
			for(int i = 0; i < l_num; i++){
				
			    int cur_id = retset[i].id;
                if(cur_id == ep_id) continue;
				
					bool good = true;
					float cur_dist = retset[i].distance;
					
					for(int j = 0; j < pp; j++){
					    if(cur_id == new_edge[j]){good = false; break;}
					
						
					    float edge_dist = fstdistfunc_(getDataByInternalId(cur_id), getDataByInternalId(new_edge[j]), dist_func_param_);	
						
						if(edge_dist < cur_dist) {good = false; break;}
						
					}                					
				
				if(good == true){
                    new_edge[pp] = cur_id;
				    pp++;
					if(pp >= maxM0_) break;
				}
				
			}
			*new_size = pp;
			
			printf("after boosting; size = %d\n", pp);
						
            visited_list_pool_->releaseVisitedList(vl);
		} 		

    };

}
