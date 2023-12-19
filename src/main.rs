use opencl3::command_queue::CommandQueue;
use opencl3::platform::get_platforms;
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};

/// N: dim
/// A: matrix
/// b: rhs
/// x: start vector
/// eps: accuracy
/// result: x
fn jacobi(dim: usize, mat: Vec<f32>, b: Vec<f32>, mut x: Vec<f32>, eps: f32) -> Vec<f32> {

    let mut y = x.clone();
   
    let mut iter = 0;
    loop {
        for i in 0..dim {
            let mut ax = 0.0;
            for k in 0..dim {
                if k != i {
                    let idx = i * dim + k;
                    ax += mat[idx] * x[k];
                }
            } 
            y[i] = (b[i] - ax) / mat[i * dim + i];
        }

        let diff: f32 = x.iter()
            .zip(y.iter())
            .map(|(x,y)| (*x-*y).abs())
            .sum();

        x = y.clone();

        log::info!("Iteration {}", iter);
        iter += 1;
        log::info!("diff: {}", diff);
        log::info!("x: {:?}", x);

        if diff < eps {
            break;
        }
    }

    x
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_jacobi() {
        env_logger::init();
        let dim = 3;
        let mat = vec![2.0, 1.0, 1.0,
                       1.0, 2.0, 1.0,
                       1.0, 1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let x = vec![0.0, 0.0, 0.0];
        let eps = 0.0001;
        let result = jacobi(dim, mat, b, x, eps);
        let expected = vec![0.5, 0.99999994, 1.5];
        assert_eq!(result, expected);
    }
}

fn main() {

    env_logger::init();

    // 1. prepare context, device and queue
    let platform = get_platforms()
            .expect("Could not query platforms.")
            .first()
            .expect("No platforms found.")
            .to_owned();


    let device_id = platform.get_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not query devices.")
            .first()
            .expect("No platforms found.")
            .to_owned();

    let device = Device::new(device_id);
    let context = Context::from_device(&device)
            .expect("Failed to create OpenCL context.");
    let _queue = CommandQueue::create_default_with_properties(&context, 0, 0);

    // 2. 
    let dim = 3;
    let mat = vec![2.0, 1.0, 1.0,
                   1.0, 2.0, 1.0,
                   1.0, 1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];
    let x = vec![0.0, 0.0, 0.0];
    let eps = 0.0001;
    let result = jacobi(dim, mat, b, x, eps);

}
