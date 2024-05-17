__global__ void calculate_sumterm_part(float2* Kerr, float2* Ra, const float2* A_t, const float* SRa, const unsigned char* SRa_nonzero_midx1234s, const float* SK, const unsigned char* SK_nonzero_midx1234s, const unsigned int N, const unsigned int M, const unsigned int NUM_SRa_NONZERO, const unsigned int NUM_SK_NONZERO, const unsigned int NUM_MODES) {
    unsigned int full_thread_idx = threadIdx.x + blockIdx.x*blockDim.x;

    // Calculate the index
    unsigned int Midx = full_thread_idx / N;
    unsigned int Nidx = full_thread_idx % N;
    unsigned int NM = N*M;

    if (full_thread_idx >= NM) {
        return;
    }

    for (int i = 0; i < NUM_SK_NONZERO; i++) {
        // Compute the SK tensors
        unsigned int SK_midx1 = SK_nonzero_midx1234s[i*4]-1;
        unsigned int SK_midx2 = SK_nonzero_midx1234s[1+i*4]-1;
        unsigned int SK_midx3 = SK_nonzero_midx1234s[2+i*4]-1;
        unsigned int SK_midx4 = SK_nonzero_midx1234s[3+i*4]-1;

        float SK_a = A_t[Nidx+Midx*N+SK_midx2*NM].x;
        float SK_b = A_t[Nidx+Midx*N+SK_midx2*NM].y;
        float SK_c = A_t[Nidx+Midx*N+SK_midx3*NM].x;
        float SK_d = A_t[Nidx+Midx*N+SK_midx3*NM].y;
        float SK_e = A_t[Nidx+Midx*N+SK_midx4*NM].x;
        float SK_f = A_t[Nidx+Midx*N+SK_midx4*NM].y;

        Kerr[Nidx+Midx*N+SK_midx1*NM].x = Kerr[Nidx+Midx*N+SK_midx1*NM].x + SK[i]*(SK_a*SK_c*SK_e-SK_b*SK_d*SK_e+SK_a*SK_d*SK_f+SK_c*SK_b*SK_f);
        Kerr[Nidx+Midx*N+SK_midx1*NM].y = Kerr[Nidx+Midx*N+SK_midx1*NM].y + SK[i]*(SK_a*SK_d*SK_e+SK_c*SK_b*SK_e-SK_a*SK_c*SK_f+SK_b*SK_d*SK_f);

        // Compute the SRa tensors
        // Because NUM_SK_NONZERO >= NUM_SRa_NONZERO, I calculate SRa tensors inside SK for-loop.
        if (i < NUM_SRa_NONZERO) {
            unsigned int SRa_midx1 = SRa_nonzero_midx1234s[i*4]-1;
            unsigned int SRa_midx2 = SRa_nonzero_midx1234s[1+i*4]-1;
            unsigned int SRa_midx3 = SRa_nonzero_midx1234s[2+i*4]-1;
            unsigned int SRa_midx4 = SRa_nonzero_midx1234s[3+i*4]-1;

            float SRa_c = A_t[Nidx+Midx*N+SRa_midx3*NM].x;
            float SRa_d = A_t[Nidx+Midx*N+SRa_midx3*NM].y;
            float SRa_e = A_t[Nidx+Midx*N+SRa_midx4*NM].x;
            float SRa_f = A_t[Nidx+Midx*N+SRa_midx4*NM].y;

            Ra[Nidx+Midx*N+SRa_midx1*NM+SRa_midx2*NM*NUM_MODES].x = Ra[Nidx+Midx*N+SRa_midx1*NM+SRa_midx2*NM*NUM_MODES].x + SRa[i]*(SRa_c*SRa_e+SRa_d*SRa_f);
            Ra[Nidx+Midx*N+SRa_midx1*NM+SRa_midx2*NM*NUM_MODES].y = Ra[Nidx+Midx*N+SRa_midx1*NM+SRa_midx2*NM*NUM_MODES].y + SRa[i]*(SRa_d*SRa_e-SRa_c*SRa_f);
        }
    }
}
