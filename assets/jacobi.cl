int next_power_of_two(int n) {
    int p = 1;
    while (p < n) {
        p = p << 1;
    }
    return p;
}

kernel void jacobi_step(
    const global uint* dim,
    const global float* mat,
    const global float* b,
    const global float* x,
    global float* y,
    global float* residuals
) {
    if (get_local_size(0) >= 128) {
        printf("ERROR: local size must be less than 128\n");
        return;
    }

    local float m_times_x[128];

    int i = get_global_id(0);
    int tx = get_local_id(0);
    int bx = get_group_id(0);

    if (tx != bx) {
        m_times_x[tx] = mat[i] * x[tx];
    }

    for (int stride = next_power_of_two(get_local_size(0)); stride > 0; stride >>= 1) {
        // sync threads in work-group
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tx < stride && tx + stride < *dim) {
            m_times_x[tx] += m_times_x[tx + stride];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (tx == 0) {
        y[bx] = (b[bx] - m_times_x[0]) / mat[bx * (*dim) + bx];
        residuals[bx] = fabs(y[bx] - x[bx]);
    }
}

kernel void residual_step(
    const global uint* dim,
    global float* residuals,
    global float* residual
) {
    int i = get_global_id(0);

    for (int stride = next_power_of_two(get_local_size(0)); stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (i < stride && i + stride < *dim) {
            residuals[i] += residuals[i + stride];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i == 0) {
        *residual = residuals[0];
    }
}
