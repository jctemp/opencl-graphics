mod jacobi;
mod opencl;

use jacobi::*;
use opencl::OclRuntime;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about)]
/// A simple OpenCL application that solves a linear system of equations.
/// It uses the Jacobi method to iteratively solve the system. The Jacobi method
/// is a simple iterative method that is guaranteed to converge if the matrix is
/// diagonally dominant.
struct Cli {
    /// Path to the OpenCL kernels. Defaults to the current working directory.
    path: Option<PathBuf>,

    /// The maximum number of iterations to perform.
    #[clap(short, long, default_value = "1000")]
    max: usize,

    /// The error tolerance.
    #[clap(short, long, default_value = "0.01")]
    eps: f32,

    /// Use the CPU instead of the GPU.
    #[clap(short, long, default_value = "false")]
    cpu: bool,
}

fn main() {
    env_logger::init();

    let args = Cli::parse();

    let path = if let Some(path) = args.path {
        std::fs::canonicalize(&path).expect("User did not provide sound path.")
    } else {
        std::fs::canonicalize(&PathBuf::from("./")).expect("Should never fail")
    };

    let file = format!("{}/jacobi.cl", path.to_str().unwrap());
    let ocl_runtime = OclRuntime::create(&file).unwrap();
    let solver = JacobiIteration::new(ocl_runtime);

    // 3. Create data
    let dim = 3;
    let mat = vec![5.0, 1.0, 1.0, 1.0, 5.0, 0.0, 1.0, 0.0, 5.0];
    let b = vec![1.0, 2.0, 0.0];
    let x = vec![0.0, 0.0, 0.0];

    let job = Job::new(
        dim,
        mat,
        b,
        x,
        args.max,
        args.eps,
        if args.cpu {
            jacobi::Mode::CPU
        } else {
            jacobi::Mode::GPU
        },
    );

    let result = solver.solve(job);

    println!("Result: {:?}", result);
}
