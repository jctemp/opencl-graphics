use bevy::math::Vec3;
use opencl3::{
    kernel::ExecuteKernel,
    memory::{CL_MAP_READ, CL_MAP_WRITE},
    svm::SvmVec,
    types::{cl_float, cl_uint, CL_BLOCKING},
};

use crate::ocl_runtime::OclRuntime;

#[derive(Debug, Clone, Copy)]
pub enum Mode {
    CPU,
    GPU,
}

#[derive(Debug)]
pub struct OclExecutor {
    ocl_runtime: OclRuntime,
}

impl OclExecutor {
    /// Create a new instance of JacobiIteration.
    pub fn new(ocl_runtime: OclRuntime) -> Self {
        Self { ocl_runtime }
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
}

//   █ ▄▀█ █▀▀ █▀█ █▄▄ █
// █▄█ █▀█ █▄▄ █▄█ █▄█ █

pub struct JacobiJob {
    dim: usize,
    mat: Vec<f32>,
    rhs: Vec<f32>,
    x: Vec<f32>,
    max: usize,
    eps: f32,
    mode: Mode,
}

impl OclExecutor {
    /// Solve the given job. Delegate to the appropriate method based on the mode.
    pub fn solve_jacobi(&self, job: JacobiJob) -> Vec<f32> {
        match job.mode {
            Mode::CPU => self.jacobi_cpu(job),
            Mode::GPU => self.jacobi_gpu(job),
        }
    }

    fn jacobi_gpu(&self, job: JacobiJob) -> Vec<f32> {
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

    fn jacobi_cpu(&self, job: JacobiJob) -> Vec<f32> {
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

// █▀▀ █▀█ █   █ █▄ █ █▀▀
// ▄▄█ █▀▀ █▄▄ █ █ ▀█ ██▄

pub struct SplineJob {
    pub samples: usize,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub max: usize,
    pub eps: f32,
    pub mode: Mode,
}

impl OclExecutor {
    /// Solve the given job. Delegate to the appropriate method based on the mode.
    pub fn solve_spline(&self, job: SplineJob) -> Vec<Vec3> {
        let x = job.x;
        let y = job.y;
        let dim = x.len();
        let mut mat = vec![0.0; dim * dim];
        let mut rhs = vec![0.0; dim];

        for i in 1..(dim - 1) {
            let mut row = vec![0.0; dim];

            row[i] = 4.0;
            row[i - 1] = 1.0;
            row[i + 1] = 1.0;

            for j in 0..dim {
                mat[i * dim + j] = row[j];
            }

            let h = x[i + 1] - x[i];
            rhs[i] = (6.0 / h) * (y[i + 1] - 2.0 * y[i] + y[i - 1]);
        }

        mat[0] = 1.0;
        mat[dim * dim - 1] = 1.0;

        rhs[0] = 0.0;
        rhs[dim - 1] = 0.0;

        log::debug!("mat: {:?}", mat);
        log::debug!("rhs: {:?}", rhs);
        log::debug!("dim: {:?}", dim);

        let mode = job.mode;
        let job = JacobiJob {
            dim,
            mat,
            rhs,
            mode,
            x: x.clone(),
            max: job.max,
            eps: job.eps,
        };

        let c = self.solve_jacobi(job);
        let mut b = vec![0.0; dim];
        let mut a = vec![0.0; dim];

        match mode {
            Mode::CPU => self.coefficients_step_cpu(&x, &y, &c, &mut b, &mut a),
            Mode::GPU => self.coefficients_step_gpu(&x, &y, &c, &mut b, &mut a),
        };

        Self::points(&x, &c, &b, &a)
    }

    fn coefficients_step_gpu(
        &self,
        x: &Vec<f32>,
        y: &Vec<f32>,
        c: &Vec<f32>,
        b: &mut Vec<f32>,
        a: &mut Vec<f32>,
    ) {
        let dim = x.len();
        // 1. Allocate memory
        let mut cl_y = SvmVec::<cl_float>::allocate(&self.ocl_runtime.context, dim).unwrap();
        let mut cl_x = SvmVec::<cl_float>::allocate(&self.ocl_runtime.context, dim).unwrap();
        let mut cl_c = SvmVec::<cl_float>::allocate(&self.ocl_runtime.context, dim).unwrap();

        // 2. Copy data to the GPU
        Self::enqueue_svm_map_write(&self.ocl_runtime, &mut cl_y);
        Self::enqueue_svm_map_write(&self.ocl_runtime, &mut cl_x);
        Self::enqueue_svm_map_write(&self.ocl_runtime, &mut cl_c);

        cl_y.clone_from_slice(y.as_slice());
        cl_x.clone_from_slice(x.as_slice());
        cl_c.clone_from_slice(c.as_slice());

        // 3. Allocate additional memory for the GPU
        let mut cl_b = SvmVec::<cl_float>::allocate(&self.ocl_runtime.context, dim).unwrap();
        let mut cl_a = SvmVec::<cl_float>::allocate(&self.ocl_runtime.context, dim).unwrap();

        // 4. Perform the computation
        let coefficients_kernel_event = unsafe {
            ExecuteKernel::new(&self.ocl_runtime.kernel_coefficients_step)
                .set_arg_svm(cl_y.as_ptr())
                .set_arg_svm(cl_x.as_ptr())
                .set_arg_svm(cl_c.as_ptr())
                .set_arg_svm(cl_b.as_mut_ptr())
                .set_arg_svm(cl_a.as_mut_ptr())
                .set_global_work_size(dim - 1)
                .enqueue_nd_range(&self.ocl_runtime.queue)
                .unwrap()
        };

        coefficients_kernel_event.wait().unwrap();

        // 5. Copy data from the GPU
        Self::enqueue_svm_map_read(&self.ocl_runtime, &mut cl_b);
        Self::enqueue_svm_map_read(&self.ocl_runtime, &mut cl_a);

        for i in 0..dim {
            b[i] = cl_b[i];
            a[i] = cl_a[i];
        }

        // 6. Release memory
        Self::enqueue_svm_unmap(&self.ocl_runtime, &mut cl_y);
        Self::enqueue_svm_unmap(&self.ocl_runtime, &mut cl_x);
        Self::enqueue_svm_unmap(&self.ocl_runtime, &mut cl_c);
        Self::enqueue_svm_unmap(&self.ocl_runtime, &mut cl_b);
        Self::enqueue_svm_unmap(&self.ocl_runtime, &mut cl_a);
    }

    fn coefficients_step_cpu(
        &self,
        x: &Vec<f32>,
        y: &Vec<f32>,
        c: &Vec<f32>,
        b: &mut Vec<f32>,
        a: &mut Vec<f32>,
    ) {
        for i in 1..x.len() {
            let h = x[i] - x[i - 1];
            b[i] = (1.0 / h) * (y[i] - y[i - 1]) - (h / 6.0) * (c[i] - c[i - 1]);
            a[i] = y[i - 1] + 0.5 * b[i] * h - (1.0 / 6.0) * c[i - 1] * h * h;
        }
    }

    fn points(x: &Vec<f32>, c: &Vec<f32>, b: &Vec<f32>, a: &Vec<f32>) -> Vec<Vec3> {
        (0..100)
            .map(|i| {
                let min = x[0];
                let max = x[x.len() - 1];
                let t = min + (max - min) * (i as f32 / 100.0);
                let y = Self::interpolate(&x, &c, &b, &a, t);
                Vec3::new(t, y, 0.0)
            })
            .collect()
    }

    fn interpolate(x: &Vec<f32>, c: &Vec<f32>, b: &Vec<f32>, a: &Vec<f32>, t: f32) -> f32 {
        let mut i = 1;
        while i < x.len() - 1 && t > x[i] {
            i += 1;
        }

        // x[i - 1] <= t <= x[i]
        let inv_h6 = 1.0 / (6.0 * (x[i] - x[i - 1]));
        let x0 = x[i - 1];
        let x1 = x[i];
        let x = t;

        inv_h6 * c[i] * (x - x0).powi(3)
            + inv_h6 * c[i - 1] * (x1 - x).powi(3)
            + b[i] * (x - 0.5 * (x0 + x1))
            + a[i]
    }
}
