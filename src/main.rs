use opencl3::command_queue::CommandQueue;
use opencl3::kernel::{Kernel, ExecuteKernel};
use opencl3::memory::{CL_MAP_WRITE, CL_MAP_READ};
use opencl3::platform::get_platforms;
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};
use opencl3::program::{Program, CL_STD_2_0};

use clap::Parser;
use opencl3::svm::SvmVec;
use opencl3::types::{cl_float, cl_uint, CL_BLOCKING};
use std::path::PathBuf;

#[derive(Parser)]
struct Cli {
    // Path to the OpenCL kernels. Defaults to working directory
    path: Option<PathBuf>,
}

struct Application {
    context: Context,
    queue: CommandQueue,
    kernel_jacobi_step: Kernel,
    kernel_residual_step: Kernel,
}

impl Application {
    fn init(directory: PathBuf) -> Self {
        // 1. prepare context, device and queue
        let platform = get_platforms()
                .expect("Could not query platforms.")
                .first()
                .expect("No platforms found.")
                .to_owned();
        log::info!("Platform: {:?}", platform.name());

        let device_id = platform.get_devices(CL_DEVICE_TYPE_GPU)
                .expect("Could not query devices.")
                .first()
                .expect("No platforms found.")
                .to_owned();
        let device = Device::new(device_id);
        log::info!("Device: {:?}", device.name());

        let context = Context::from_device(&device)
                .expect("Failed to create OpenCL context.");
        let queue = CommandQueue::create_default_with_properties(&context, 0, 0)
                .expect("Failed to create CommandQueue.");

        // 2. Create program
        let file =  format!("{}/jacobi.cl", directory.to_str().expect("Cannot convert path to str."));
        log::info!("Target: {}", file);

        let kernel_source = std::fs::read_to_string(file)
            .expect("Failed to parse file.");
        let program = Program::create_and_build_from_source(&context, &kernel_source, CL_STD_2_0)
            .expect("Failed to build program.");
        let kernel_jacobi_step = Kernel::create(&program, "jacobi_step")
            .expect("Kernel::create failed");
        let kernel_residual_step = Kernel::create(&program, "residual_step")
            .expect("Kernel::create failed");

        Application {
            context, queue, kernel_jacobi_step, kernel_residual_step
        }
    }
}

fn jacobi_step(dim: usize, mat: &[f32], b: &[f32], x: &[f32], y: &mut [f32], residuals: &mut [f32]) {
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

fn main() {
    env_logger::init();

    let args = Cli::parse();

    let path = if let Some(path) = args.path {
        std::fs::canonicalize(&path)
            .expect("User did not provide sound path.")
    } else {
        std::fs::canonicalize(&PathBuf::from("./"))
            .expect("Should never fail")
    };

    log::info!("OpenCL file directory {:?}", path);


    // 3. Create data
    let dim = 3;
    let mat = vec![5.0, 1.0, 1.0,
                   1.0, 5.0, 0.0,
                   1.0, 0.0, 5.0];
    let b = vec![1.0, 2.0, 0.0];
    let mut x = vec![0.0, 0.0, 0.0];
    let eps = 0.01;
    let max = 1000;

    let app = Application::init(path);

    let mut cl_dim = SvmVec::<cl_uint>::allocate(&app.context, 1).unwrap();
    let mut cl_mat = SvmVec::<cl_float>::allocate(&app.context, dim * dim).unwrap();
    let mut cl_b = SvmVec::<cl_float>::allocate(&app.context, dim).unwrap();
    let mut cl_x = SvmVec::<cl_float>::allocate(&app.context, dim).unwrap();
    let mut cl_max = SvmVec::<cl_uint>::allocate(&app.context, 1).unwrap();

    if !cl_dim.is_fine_grained() {
        unsafe {
            app.queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut cl_dim, &[]).unwrap();
            app.queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut cl_mat, &[]).unwrap();
            app.queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut cl_b, &[]).unwrap();
            app.queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut cl_x, &[]).unwrap();
            app.queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut cl_max, &[]).unwrap();
        }
    }

    cl_dim.clone_from_slice(&[dim as u32]);
    cl_mat.clone_from_slice(mat.as_slice());
    cl_b.clone_from_slice(b.as_slice());
    cl_x.clone_from_slice(x.as_slice());
    cl_max.clone_from_slice(&[max]);

    let mut cl_y = SvmVec::<cl_float>::allocate(&app.context, dim).unwrap();
    let mut cl_residuals= SvmVec::<cl_float>::allocate(&app.context, dim).unwrap();
    let mut cl_residual= SvmVec::<cl_float>::allocate(&app.context, 1).unwrap();
    cl_residual[0] = f32::INFINITY;

    let mut y = [0.0;3];
    let mut residuals = [0.0;3];
    let mut residual = [f32::INFINITY];

    for i in 0..6 {
        
        jacobi_step(dim, mat.as_slice(), b.as_slice(), x.as_slice(), y.as_mut_slice(), residuals.as_mut_slice());

        let jacobi_kernel_event = unsafe {
            ExecuteKernel::new(&app.kernel_jacobi_step)
                .set_arg_svm(cl_dim.as_ptr())
                .set_arg_svm(cl_mat.as_ptr())
                .set_arg_svm(cl_b.as_ptr())
                .set_arg_svm(cl_x.as_ptr())
                .set_arg_svm(cl_y.as_mut_ptr())
                .set_arg_svm(cl_residuals.as_mut_ptr())
                .set_local_work_size(dim)
                .set_global_work_size(dim*dim)
                .enqueue_nd_range(&app.queue)
                .unwrap()
        };

        jacobi_kernel_event.wait().unwrap();

        jacobi_step(dim, mat.as_slice(), b.as_slice(), y.as_slice(), x.as_mut_slice(), residuals.as_mut_slice());
        
        let jacobi_kernel_event = unsafe {
            ExecuteKernel::new(&app.kernel_jacobi_step)
                .set_arg_svm(cl_dim.as_ptr())
                .set_arg_svm(cl_mat.as_ptr())
                .set_arg_svm(cl_b.as_ptr())
                .set_arg_svm(cl_y.as_ptr())
                .set_arg_svm(cl_x.as_mut_ptr())
                .set_arg_svm(cl_residuals.as_mut_ptr())
                .set_local_work_size(dim)
                .set_global_work_size(dim*dim)
                .enqueue_nd_range(&app.queue)
                .unwrap()
        };

        jacobi_kernel_event.wait().unwrap();

        residual_step(dim, residuals.as_mut_slice(), residual.as_mut_slice());

        let residual_kernel_event = unsafe {
            ExecuteKernel::new(&app.kernel_residual_step)
                .set_arg_svm(cl_dim.as_ptr())
                .set_arg_svm(cl_residuals.as_mut_ptr())
                .set_arg_svm(cl_residual.as_mut_ptr())
                .set_global_work_size(dim)
                .enqueue_nd_range(&app.queue)
                .unwrap()
        };

        residual_kernel_event.wait().unwrap();

        if !cl_residual.is_fine_grained() {
            unsafe {
                app.queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_READ, &mut cl_residual, &[]).unwrap();
                app.queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_READ, &mut cl_y, &[]).unwrap();
            }
        }

        log::info!("GPU, CPU - Iteration {}", i * 2);
        log::info!("{:?}, {:?}", cl_x, x);
        log::info!("{:?}, {:?}", cl_residuals, residuals);
        log::info!("{:?}, {:?}", cl_residual, residual);
    }

    if !cl_y.is_fine_grained() {
        let unmap_results_event = unsafe { app.queue.enqueue_svm_unmap(&cl_y, &[]).unwrap() };
        unmap_results_event.wait().unwrap();
    }
}
