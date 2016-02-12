#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

#include <mpi.h>

//We do a small sieve so it fits in L1 Cache
#define SV_SIZE 120000
int64_t count_primes(int64_t n_min, int64_t n_max, int *p_cache, int cache_sz);
int *sv_primes(int n_max, int *p_count);

int mpi_rank;
int mpi_comm_sz;

int main(int argc, char *argv[]) 
{
    //Read in the maximum number
    int64_t n_max;
    sscanf(argv[1], "%lld", &n_max);
    
    //MPI Setup
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_sz);
    
    int *p_cache;
    int p_cache_sz;
    
    //Set up pairs of n_min/n_max for each process
    int64_t n_bounds[3 * mpi_comm_sz];
    int64_t bound_sz = n_max / mpi_comm_sz;
    //bound_sz += bound_sz % 2;
    if (mpi_rank == 0) {
        
        //Each process will sieve over the interval
        //[b_min, b_max]
        for (int j = 0; j < mpi_comm_sz; j++) {
        
            int64_t *bounds = &n_bounds[3 * j];
        
            *(bounds) = j * bound_sz;
            *(bounds + 1) = (j + 1) * bound_sz - 1;
            *(bounds + 2) = bound_sz - 1;

            //Give extra numbers to the last process
            if (j == mpi_comm_sz - 1) {
                *(bounds + 1) = n_max;
                *(bounds + 2) = n_max - *(bounds + 1);
            }
        }
        
        printf("threads: %d n_max %lld\n", mpi_comm_sz, n_max);
            
        p_cache = sv_primes((int) (sqrt(n_max) + .5), &p_cache_sz);
    }
    
    //Give each process an interval to sieve over
    int64_t _sv_ival[3];
    MPI_Scatter(n_bounds, 3, MPI_LONG_LONG_INT, _sv_ival, 3,
                    MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
                    
    //Send out our cached primes, as well as how many have been cached
    MPI_Bcast(&p_cache_sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (mpi_rank != 0) {
        p_cache = malloc(p_cache_sz * sizeof(int));
    }
    
    MPI_Bcast(p_cache, p_cache_sz, MPI_INT, 0, MPI_COMM_WORLD);
    
    int64_t _ival_min = _sv_ival[0];
    int64_t _ival_max = _sv_ival[1];
    int64_t _ival_sz = _sv_ival[2];
    
    
    int64_t _result = count_primes(_ival_min, _ival_max, p_cache, p_cache_sz);
    
    //Sum up the counts from each process
    int64_t final_result;
    MPI_Reduce(&_result, &final_result, 1, MPI_LONG_LONG_INT, MPI_SUM, 0,
                    MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_rank == 0) {               
        //Add one (since 2 is a prime)
        printf("result: %lld\n", final_result);
    }
        
    MPI_Finalize();
    free(p_cache);
    return 0;
}

int64_t count_primes(int64_t n_min, int64_t n_max, int *p_cache, int p_cache_sz)
{
    int64_t p_count = 0;
    
    int64_t block_sz = SV_SIZE;
    int64_t block_count = (int64_t) ((n_max - n_min) / SV_SIZE + 1);
    
    int64_t curr_block = 0;
    int64_t block_start = n_min;
    
    int64_t block_end = (block_start + SV_SIZE) < n_max ? block_start + SV_SIZE : n_max;
    int64_t n = n_min;
        
    //We're not actually storing the values, so we don't need to use int64_t
    char *sieve_buf = malloc((SV_SIZE + 1) * sizeof(char));
    memset(sieve_buf, 1, (SV_SIZE + 1) * sizeof(char) + 1);
    
    printf("rank %d: n_min: %lld n_max: %lld block_count: %lld block_sz: %lld\n", mpi_rank,
                    n_min, n_max, block_count, block_sz);
                    
    while (block_start < n_max) {
    
        //printf("start: %lld end: %lld\n", block_start, block_end);
        int block_p_count = 0;
        //The number of integers we are searching
        curr_block++;

        if (block_start == 0) {
            sieve_buf[0] = 0;
            sieve_buf[1] = 0;
            sieve_buf[2] = 1;
        }
        
        //Mark multiples of primes in the cache as being composite
        for (int i = 0; i < p_cache_sz; i++) {
            int p = p_cache[i];
            
            //Find the first multiple of p within the block
            int64_t p_mult = p * (block_start / p);
           
            //If block_start < p, then p_mult = p and thus p would be marked
            //as composite. Make sure p_mult > p
            while (p_mult <= p) {
                p_mult += p;
            }
            
            while (p_mult < block_start) {
                p_mult += p;
            }
            
            while (p_mult <= block_end) {
                sieve_buf[(p_mult - block_start)] = 0;      
                //Move to the next multiple of p     
                p_mult += p; 
            }
        }
          
        //count the primes in the current block
        for (int j = 0; j <= SV_SIZE; j++) {

            if (block_start + j == n_max) {
                block_p_count += sieve_buf[j];
                break;
            }
            
            //Don't count everything if we're the last block
            block_p_count += sieve_buf[j];
        }         
        block_start = block_end + 1;
        block_end = (block_start + block_sz) < n_max ? block_start + block_sz : n_max;
        p_count += block_p_count;
        
        memset(sieve_buf, 1, (SV_SIZE + 1) * sizeof(char));
    }   
    
    free(sieve_buf);
    
    return p_count;
}

int *sv_primes(int n_max, int *p_count)
{
    int num_p = 1;
    
    int *sieve_buf = malloc(n_max * sizeof(int));
    memset(sieve_buf, 1, n_max * sizeof(int));
    
    sieve_buf[0] = 2;
    int *next_p = &sieve_buf[1];

    for (int n = 3; n <= n_max; n += 2) {
        //The number is composite
        if (sieve_buf[n] == 0) {
            continue;
        }
        
        int k = 1;
        while (n * k < n_max) {
            sieve_buf[n * k] = 0;
            k++;
        }
        
        *next_p = n;
        next_p++;
        num_p++;
    }
    
    *p_count = num_p;
    
    return sieve_buf;
}        
        

