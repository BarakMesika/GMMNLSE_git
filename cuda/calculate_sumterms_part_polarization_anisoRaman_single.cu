__global__ void calculate_sumterm_part(float2 * Kerr, float2 * Ra, float2 * Rb, float2 * transfer_matrix, const float2 * A_t, const float2 * Bmn, const float* SRa, const unsigned char* SRa_nonzero_midx1234s, const float* SRb, const unsigned char* SRb_nonzero_midx1234s, const float* SK, const unsigned char* SK_nonzero_midx1234s, const unsigned int N, const unsigned int M, const unsigned int NUM_SRa_NONZERO, const unsigned int NUM_SK_NONZERO, const unsigned int NUM_MODES) {
    unsigned int full_thread_idx = threadIdx.x + blockIdx.x*blockDim.x;

    // Calculate the index
    unsigned int Midx = full_thread_idx / N;
    unsigned int Nidx = full_thread_idx % N;
    unsigned int NM = N*M;

    unsigned int NUM_SPATIAL_MODES = NUM_MODES / 2;
    unsigned int square = NUM_SPATIAL_MODES*NUM_SPATIAL_MODES;

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

        // Compute the SRb tensors
        unsigned int SRb_midx1 = SRb_nonzero_midx1234s[i*4]-1;
        unsigned int SRb_midx2 = SRb_nonzero_midx1234s[1+i*4]-1;
        unsigned int SRb_midx3 = SRb_nonzero_midx1234s[2+i*4]-1;
        unsigned int SRb_midx4 = SRb_nonzero_midx1234s[3+i*4]-1;

        float SRb_c = A_t[Nidx+Midx*N+SRb_midx3*NM].x;
        float SRb_d = A_t[Nidx+Midx*N+SRb_midx3*NM].y;
        float SRb_e = A_t[Nidx+Midx*N+SRb_midx4*NM].x;
        float SRb_f = A_t[Nidx+Midx*N+SRb_midx4*NM].y;

        Rb[Nidx+Midx*N+SRb_midx1*NM+SRb_midx2*NM*NUM_MODES].x = Rb[Nidx+Midx*N+SRb_midx1*NM+SRb_midx2*NM*NUM_MODES].x + SRb[i]*(SRb_c*SRb_e+SRb_d*SRb_f);
        Rb[Nidx+Midx*N+SRb_midx1*NM+SRb_midx2*NM*NUM_MODES].y = Rb[Nidx+Midx*N+SRb_midx1*NM+SRb_midx2*NM*NUM_MODES].y + SRb[i]*(SRb_d*SRb_e-SRb_c*SRb_f);

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

            // Do the calculation for the transfer matrix
            if (Nidx == 0) { // "Nidx==0" is to limit this to run only once through different Nidx
                // the index related to polarizations
                unsigned int polar_idx_T = SRa_midx1 % 2;
                unsigned int polar_idx_B = SRa_midx3 % 2;

                // recovery from total indices back to spatial mode indices
                unsigned int spatial_midx1 = SRa_midx1 / 2;
                unsigned int spatial_midx2 = SRa_midx2 / 2;
                unsigned int spatial_midx3 = SRa_midx3 / 2;
                unsigned int spatial_midx4 = SRa_midx4 / 2;

                // the dimension of the computation/assignment below
                unsigned int idx2_T = spatial_midx1*M;
                unsigned int idx3_T = spatial_midx2*M*NUM_SPATIAL_MODES;
                unsigned int idx4_T = polar_idx_T*M*square;

                unsigned int idx2_B = spatial_midx3*M;
                unsigned int idx3_B = spatial_midx4*M*NUM_SPATIAL_MODES;
                unsigned int idx4_B = polar_idx_B*M*square;

                transfer_matrix[Midx+idx2_T+idx3_T+idx4_T].x = transfer_matrix[Midx+idx2_T+idx3_T+idx4_T].x + SRa[i]*Bmn[Midx+idx2_B+idx3_B+idx4_B].x;
                transfer_matrix[Midx+idx2_T+idx3_T+idx4_T].y = transfer_matrix[Midx+idx2_T+idx3_T+idx4_T].y + SRa[i]*Bmn[Midx+idx2_B+idx3_B+idx4_B].y;
            }
        }
    }
}
