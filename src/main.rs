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

    // Cubic spline interpolation

    let x = vec![0.0, 2.0, 4.0, 6.0, 8.0];
    let y = vec![0.0, 1.0, 0.0, 1.0, 0.0];

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

    let job = Job::new(
        dim,
        mat,
        rhs,
        x.clone(),
        args.max,
        args.eps,
        if args.cpu {
            jacobi::Mode::CPU
        } else {
            jacobi::Mode::GPU
        },
    );

    let c = solver.solve(job);

    println!("c's: {:?}", c);

    let mut b = vec![0.0; dim];
    for i in 1..dim {
        let h = x[i] - x[i - 1];
        b[i] = (1.0 / h) * (y[i] - y[i - 1]) - (h / 6.0) * (c[i] - c[i - 1]);
    }

    println!("b's: {:?}", b);

    let mut a = vec![0.0; dim];
    for i in 1..dim {
        let h = x[i] - x[i - 1];
        a[i] = y[i - 1] + (1.0 / 2.0) * b[i] * h - (1.0 / 6.0) * c[i - 1] * h * h;
    }

    println!("a's: {:?}", a);

    let mut s = vec![0.0; dim];
    let arg = 2.0;
    for i in 1..dim {
        let h = x[i] - x[i - 1];
        s[i] = (1.0 / (6.0 * h)) * c[i] * (arg - x[i - 1]).powi(3)
            + (1.0 / (6.0 * h)) * c[i - 1] * (x[i] - arg)
            + b[i] * (arg - 0.5 * (x[i - 1] + x[i]))
            + a[i];
    }
    println!("s: {:?}", s);
    let sum = s.iter().sum::<f32>();

    println!("s: {:?}", sum);
}
