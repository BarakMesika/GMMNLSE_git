__global__ void calculate_sumterm_part(double2 * Up, double2 * Vpl, double2 * transfer_matrix, const double2 * A_t, const double2 * Bmn, const double* SR, const unsigned char* SR_nonzero_midx1234s, const double* SK, const unsigned char* SK_nonzero_midx1234s, const unsigned int N, const unsigned int M, const unsigned int NUM_SR_NONZERO, const unsigned int NUM_SK_NONZERO, const unsigned int NUM_MODES) {
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
        unsigned int midx1 = SK_nonzero_midx1234s[i*4]-1;
        unsigned int midx2 = SK_nonzero_midx1234s[1+i*4]-1;
        unsigned int midx3 = SK_nonzero_midx1234s[2+i*4]-1;
        unsigned int midx4 = SK_nonzero_midx1234s[3+i*4]-1;

        double a = A_t[Nidx+Midx*N+midx2*NM].x;
        double b = A_t[Nidx+Midx*N+midx2*NM].y;
        double c = A_t[Nidx+Midx*N+midx3*NM].x;
        double d = A_t[Nidx+Midx*N+midx3*NM].y;
        double e = A_t[Nidx+Midx*N+midx4*NM].x;
        double f = A_t[Nidx+Midx*N+midx4*NM].y;

        Up[Nidx+Midx*N+midx1*NM].x = Up[Nidx+Midx*N+midx1*NM].x + SK[i]*(a*c*e-b*d*e+a*d*f+c*b*f);
        Up[Nidx+Midx*N+midx1*NM].y = Up[Nidx+Midx*N+midx1*NM].y + SK[i]*(a*d*e+c*b*e-a*c*f+b*d*f);

        // Compute the SR tensors
        // Because NUM_SK_NONZERO >= NUM_SR_NONZERO, I calculate SR tensors inside SK for-loop.
        if (i < NUM_SR_NONZERO) {
            unsigned int midx1 = SR_nonzero_midx1234s[i*4]-1;
            unsigned int midx2 = SR_nonzero_midx1234s[1+i*4]-1;
            unsigned int midx3 = SR_nonzero_midx1234s[2+i*4]-1;
            unsigned int midx4 = SR_nonzero_midx1234s[3+i*4]-1;

            double c = A_t[Nidx+Midx*N+midx3*NM].x;
            double d = A_t[Nidx+Midx*N+midx3*NM].y;
            double e = A_t[Nidx+Midx*N+midx4*NM].x;
            double f = A_t[Nidx+Midx*N+midx4*NM].y;

            Vpl[Nidx+Midx*N+midx1*NM+midx2*NM*NUM_MODES].x = Vpl[Nidx+Midx*N+midx1*NM+midx2*NM*NUM_MODES].x + SR[i]*(c*e+d*f);
            Vpl[Nidx+Midx*N+midx1*NM+midx2*NM*NUM_MODES].y = Vpl[Nidx+Midx*N+midx1*NM+midx2*NM*NUM_MODES].y + SR[i]*(d*e-c*f);

            // Do the calculation for the transfer matrix
            if (Nidx == 0) { // "Nidx==0" is to limit this to run only once through different Nidx
                // the index related to polarizations
                unsigned int polar_idx_T = midx1 % 2;
                unsigned int polar_idx_B = midx3 % 2;

                // recovery from total indices back to spatial mode indices
                unsigned int spatial_midx1 = midx1 / 2;
                unsigned int spatial_midx2 = midx2 / 2;
                unsigned int spatial_midx3 = midx3 / 2;
                unsigned int spatial_midx4 = midx4 / 2;

                // the dimension of the computation/assignment below
                unsigned int idx2_T = spatial_midx1*M;
                unsigned int idx3_T = spatial_midx2*M*NUM_SPATIAL_MODES;
                unsigned int idx4_T = polar_idx_T*M*square;

                unsigned int idx2_B = spatial_midx3*M;
                unsigned int idx3_B = spatial_midx4*M*NUM_SPATIAL_MODES;
                unsigned int idx4_B = polar_idx_B*M*square;

                transfer_matrix[Midx+idx2_T+idx3_T+idx4_T].x = transfer_matrix[Midx+idx2_T+idx3_T+idx4_T].x + SR[i]*Bmn[Midx+idx2_B+idx3_B+idx4_B].x;
                transfer_matrix[Midx+idx2_T+idx3_T+idx4_T].y = transfer_matrix[Midx+idx2_T+idx3_T+idx4_T].y + SR[i]*Bmn[Midx+idx2_B+idx3_B+idx4_B].y;
            }
        }
    }
}
