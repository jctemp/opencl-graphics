use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::Kernel;
use opencl3::platform::get_platforms;
use opencl3::program::{Program, CL_STD_2_0};

#[derive(Debug)]
pub struct OclRuntime {
    pub context: Context,
    pub queue: CommandQueue,
    pub kernel_jacobi_step: Kernel,
    pub kernel_residual_step: Kernel,
}

#[derive(Debug)]
pub enum ErrorType {
    QueryFailed(String),
    PlatformNotFound(String),
    DeviceNotFound(String),
    ContextCreationFailed(String),
    CommandQueueCreationFailed(String),
    ProgramCreationFailed(String),
    KernelCreationFailed(String),
}

#[derive(Debug)]
pub struct OclError(ErrorType);

impl std::fmt::Display for OclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match &self.0 {
            ErrorType::QueryFailed(msg) => {
                format!("Failed to query OpenCL platform or device. {}", msg)
            }
            ErrorType::PlatformNotFound(msg) => format!("No OpenCL platform found. {}", msg),
            ErrorType::DeviceNotFound(msg) => format!("No OpenCL device found. {}", msg),
            ErrorType::ContextCreationFailed(msg) => {
                format!("Failed to create OpenCL context. {}", msg)
            }
            ErrorType::CommandQueueCreationFailed(msg) => {
                format!("Failed to create OpenCL command queue. {}", msg)
            }
            ErrorType::ProgramCreationFailed(msg) => {
                format!("Failed to create OpenCL program. {}", msg)
            }
            ErrorType::KernelCreationFailed(msg) => {
                format!("Failed to create OpenCL kernel. {}", msg)
            }
        };
        write!(f, "{}", msg)
    }
}

impl std::error::Error for OclError {}

type Result<T> = std::result::Result<T, OclError>;

impl OclRuntime {
    pub fn build(kernel_source: &str) -> Result<Self> {
        // 1. prepare context, device and queue
        let platform = get_platforms()
            .map_err(|e| OclError(ErrorType::QueryFailed(e.to_string())))?
            .first()
            .ok_or(OclError(ErrorType::PlatformNotFound(String::from(
                "Count = 0",
            ))))?
            .to_owned();

        let device_id = platform
            .get_devices(CL_DEVICE_TYPE_GPU)
            .map_err(|e| OclError(ErrorType::QueryFailed(e.to_string())))?
            .first()
            .ok_or(OclError(ErrorType::DeviceNotFound(String::from(
                "GPU not detected.",
            ))))?
            .to_owned();
        let device = Device::new(device_id);

        let context = Context::from_device(&device)
            .map_err(|e| OclError(ErrorType::ContextCreationFailed(e.to_string())))?;
        let queue = CommandQueue::create_default_with_properties(&context, 0, 0)
            .map_err(|e| OclError(ErrorType::CommandQueueCreationFailed(e.to_string())))?;

        // 2. Create program
        let program = Program::create_and_build_from_source(&context, &kernel_source, CL_STD_2_0)
            .map_err(|e| OclError(ErrorType::ProgramCreationFailed(e.to_string())))?;
        let kernel_jacobi_step = Kernel::create(&program, "jacobi_step")
            .map_err(|e| OclError(ErrorType::KernelCreationFailed(e.to_string())))?;
        let kernel_residual_step = Kernel::create(&program, "residual_step")
            .map_err(|e| OclError(ErrorType::KernelCreationFailed(e.to_string())))?;

        println!("OpenCL Summary:");
        println!("  Platform: {}", platform.name().unwrap());
        println!("  Device: {}", device.name().unwrap());
        println!("  Kernel names: {}", program.get_kernel_names().unwrap());
        println!();

        Ok(Self {
            context,
            queue,
            kernel_jacobi_step,
            kernel_residual_step,
        })
    }
}
