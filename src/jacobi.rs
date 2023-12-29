use opencl3::{
    kernel::ExecuteKernel,
    memory::{CL_MAP_READ, CL_MAP_WRITE},
    svm::SvmVec,
    types::{cl_float, cl_uint, CL_BLOCKING},
};

use crate::opencl::OclRuntime;

pub enum Mode {
    CPU,
    GPU,
}

pub struct Job {
    dim: usize,
    mat: Vec<f32>,
    rhs: Vec<f32>,
    x: Vec<f32>,
    max: usize,
    eps: f32,
    mode: Mode,
}

impl Job {
    pub fn new(
        dim: usize,
        mat: Vec<f32>,
        rhs: Vec<f32>,
        x: Vec<f32>,
        max: usize,
        eps: f32,
        mode: Mode,
    ) -> Self {
        Self {
            dim,
            mat,
            rhs,
            x,
            max,
            eps,
            mode,
        }
    }
}

pub struct JacobiSolver {
    ocl_runtime: OclRuntime,
}

impl JacobiSolver {
    /// Create a new instance of JacobiIteration.
    pub fn new(ocl_runtime: OclRuntime) -> Self {
        Self { ocl_runtime }
    }

    /// Solve the given job. Delegate to the appropriate method based on the mode.
    pub fn solve(&self, job: Job) -> Vec<f32> {
        match job.mode {
            Mode::CPU => self.solve_cpu(job),
            Mode::GPU => self.solve_gpu(job),
        }
    }

    /// Set up vector to copy data to the GPU.
    fn enqueue_svm_map_write<T>(ocl_runtime: &OclRuntime, cl_vec: &mut SvmVec<T>) {
        if cl_vec.is_fine_grained() {
            unsafe {
                ocl_runtime
                    .queue
                    .enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, cl_vec, &[])
                    .unwrap();
            }
        }
    }

    /// Set up vector to copy data from the GPU.
    fn enqueue_svm_map_read<T>(ocl_runtime: &OclRuntime, cl_vec: &mut SvmVec<T>) {
        if !cl_vec.is_fine_grained() {
            unsafe {
                ocl_runtime
                    .queue
                    .enqueue_svm_map(CL_BLOCKING, CL_MAP_READ, cl_vec, &[])
                    .unwrap();
            }
        }
    }

    /// Release vector from the GPU.
    fn enqueue_svm_unmap<T>(ocl_runtime: &OclRuntime, cl_vec: &mut SvmVec<T>) {
        if cl_vec.is_fine_grained() {
            unsafe {
                ocl_runtime.queue.enqueue_svm_unmap(cl_vec, &[]).unwrap();
            }
        }
    }

    /// Solve the given job on the GPU.
    fn solve_gpu(&self, job: Job) -> Vec<f32> {
        // 1. Allocate memory
        let mut cl_dim = SvmVec::<cl_uint>::allocate(&self.ocl_runtime.context, 1).unwrap();
        let mut cl_mat =
            SvmVec::<cl_float>::allocate(&self.ocl_runtime.context, job.dim * job.dim).unwrap();
        let mut cl_b = SvmVec::<cl_float>::allocate(&self.ocl_runtime.context, job.dim).unwrap();
        let mut cl_x = SvmVec::<cl_float>::allocate(&self.ocl_runtime.context, job.dim).unwrap();

        // 2. Copy data to the GPU
        Self::enqueue_svm_map_write(&self.ocl_runtime, &mut cl_dim);
        Self::enqueue_svm_map_write(&self.ocl_runtime, &mut cl_mat);
        Self::enqueue_svm_map_write(&self.ocl_runtime, &mut cl_b);
        Self::enqueue_svm_map_write(&self.ocl_runtime, &mut cl_x);

        cl_dim.clone_from_slice(&[job.dim as u32]);
        cl_mat.clone_from_slice(job.mat.as_slice());
        cl_b.clone_from_slice(job.rhs.as_slice());
        cl_x.clone_from_slice(job.x.as_slice());

        // 3. Allocate additional memory for the GPU
        let mut cl_y = SvmVec::<cl_float>::allocate(&self.ocl_runtime.context, job.dim).unwrap();
        let mut cl_residuals =
            SvmVec::<cl_float>::allocate(&self.ocl_runtime.context, job.dim).unwrap();
        let mut cl_residual = SvmVec::<cl_float>::allocate(&self.ocl_runtime.context, 1).unwrap();
        cl_residual[0] = f32::INFINITY;

        // 4. Perform the computation
        for _ in 0..(job.max / 2) {
            let jacobi_kernel_event = unsafe {
                ExecuteKernel::new(&self.ocl_runtime.kernel_jacobi_step)
                    .set_arg_svm(cl_dim.as_ptr())
                    .set_arg_svm(cl_mat.as_ptr())
                    .set_arg_svm(cl_b.as_ptr())
                    .set_arg_svm(cl_x.as_ptr())
                    .set_arg_svm(cl_y.as_mut_ptr())
                    .set_arg_svm(cl_residuals.as_mut_ptr())
                    .set_local_work_size(job.dim)
                    .set_global_work_size(job.dim * job.dim)
                    .enqueue_nd_range(&self.ocl_runtime.queue)
                    .unwrap()
            };

            jacobi_kernel_event.wait().unwrap();

            let jacobi_kernel_event = unsafe {
                ExecuteKernel::new(&self.ocl_runtime.kernel_jacobi_step)
                    .set_arg_svm(cl_dim.as_ptr())
                    .set_arg_svm(cl_mat.as_ptr())
                    .set_arg_svm(cl_b.as_ptr())
                    .set_arg_svm(cl_y.as_ptr())
                    .set_arg_svm(cl_x.as_mut_ptr())
                    .set_arg_svm(cl_residuals.as_mut_ptr())
                    .set_local_work_size(job.dim)
                    .set_global_work_size(job.dim * job.dim)
                    .enqueue_nd_range(&self.ocl_runtime.queue)
                    .unwrap()
            };

            jacobi_kernel_event.wait().unwrap();

            let residual_kernel_event = unsafe {
                ExecuteKernel::new(&self.ocl_runtime.kernel_residual_step)
                    .set_arg_svm(cl_dim.as_ptr())
                    .set_arg_svm(cl_residuals.as_mut_ptr())
                    .set_arg_svm(cl_residual.as_mut_ptr())
                    .set_global_work_size(job.dim)
                    .enqueue_nd_range(&self.ocl_runtime.queue)
                    .unwrap()
            };

            residual_kernel_event.wait().unwrap();

            Self::enqueue_svm_map_read(&self.ocl_runtime, &mut cl_residual);

            if cl_residual[0] < job.eps {
                break;
            }
        }

        // 5. Copy data from the GPU
        Self::enqueue_svm_map_read(&self.ocl_runtime, &mut cl_x);
        let result = cl_x.to_vec();

        log::debug!("GPU - Iteration {}", job.max);
        log::debug!("x: {:?}", result);
        log::debug!("residuals: {:?}", cl_residuals.to_vec());
        log::debug!("residual: {:?}", cl_residual.to_vec());

        // 6. Release memory
        Self::enqueue_svm_unmap(&self.ocl_runtime, &mut cl_dim);
        Self::enqueue_svm_unmap(&self.ocl_runtime, &mut cl_mat);
        Self::enqueue_svm_unmap(&self.ocl_runtime, &mut cl_b);
        Self::enqueue_svm_unmap(&self.ocl_runtime, &mut cl_x);
        Self::enqueue_svm_unmap(&self.ocl_runtime, &mut cl_y);
        Self::enqueue_svm_unmap(&self.ocl_runtime, &mut cl_residuals);
        Self::enqueue_svm_unmap(&self.ocl_runtime, &mut cl_residual);

        result
    }

    fn solve_cpu(&self, job: Job) -> Vec<f32> {
        let mut x = job.x;
        let mut y = vec![0.0; job.dim];
        let mut residuals = vec![0.0; job.dim];
        let mut residual = vec![0.0; 1];

        for _ in 0..(job.max / 2) {
            Self::jacobi_step(job.dim, &job.mat, &job.rhs, &x, &mut y, &mut residuals);
            Self::jacobi_step(job.dim, &job.mat, &job.rhs, &y, &mut x, &mut residuals);
            Self::residual_step(job.dim, &mut residuals, &mut residual);

            if residual[0] < job.eps {
                break;
            }
        }

        log::debug!("CPU - Iteration {}", job.max);
        log::debug!("x: {:?}", x);
        log::debug!("residuals: {:?}", residuals);
        log::debug!("residual: {:?}", residual);

        x
    }

    fn jacobi_step(
        dim: usize,
        mat: &[f32],
        b: &[f32],
        x: &[f32],
        y: &mut [f32],
        residuals: &mut [f32],
    ) {
        for i in 0..dim {
            let mut sum = 0.0;
            for j in 0..dim {
                if i != j {
                    sum += mat[i * dim + j] * x[j];
                }
            }
            y[i] = (b[i] - sum) / mat[i * dim + i];
            residuals[i] = f32::abs(y[i] - x[i]);
        }
    }

    fn residual_step(_dim: usize, residuals: &mut [f32], residual: &mut [f32]) {
        residual[0] = residuals.iter().sum();
    }
}
