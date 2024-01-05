
mod ocl_executor;
mod ocl_runtime;

use ocl_executor::*;
use ocl_runtime::*;

use clap::Parser;
use std::{path::PathBuf, sync::{Arc, Mutex}, f32::consts::PI};

use bevy::{
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    reflect::TypePath,
    render::{
        mesh::{MeshVertexBufferLayout, PrimitiveTopology},
        render_resource::{
            AsBindGroup, PolygonMode, RenderPipelineDescriptor, ShaderRef,
            SpecializedMeshPipelineError,
        },
    },
};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Windowing
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(MaterialPlugin::<LineMaterial>::default())
        .add_systems(Startup, setup_camera)
        .add_systems(Startup, setup_line)
        .add_systems(Startup, setup_solver)
        .add_systems(FixedUpdate, update_line)
        .insert_resource(Time::<Fixed>::from_seconds(1.0))
        .run();

    Ok(())
}

#[derive(Debug, Clone, Component)]
struct Solver {
    solver: Arc<Mutex<OclExecutor>>,
    eps: f32,
    max: usize,
    mode: ocl_executor::Mode,
}

fn setup_solver(mut commands: Commands) {
    let args = Cli::parse();

    let path = if let Some(path) = args.path {
        std::fs::canonicalize(&path).expect("User did not provide sound path.")
    } else {
        std::fs::canonicalize(&PathBuf::from("./")).expect("Should never fail")
    };

    let kernel_source = std::fs::read_to_string(path).expect("Cannot read kernel source");
    let ocl_runtime = OclRuntime::build(&kernel_source).expect("Cannot build OCL runtime");
    let solver = OclExecutor::new(ocl_runtime);

    commands.spawn(Solver {
        solver: Arc::new(Mutex::new(solver)),
        eps: args.eps,
        max: args.max,
        mode: if args.cpu {
            ocl_executor::Mode::CPU
        } else {
            ocl_executor::Mode::GPU
        },
    });
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
}

fn setup_line(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<LineMaterial>>,
) {
    let line = LineStrip {
        points: vec![
            Vec3::new(-2.0, 0.0, 0.0), 
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ],
    };

    commands.spawn(line.clone());

    // Spawn a line strip that goes from point to point
    commands.spawn(MaterialMeshBundle {
        mesh: meshes.add(Mesh::from(line)),
        material: materials.add(LineMaterial {
            color: Color::rgb(1.0, 0.0, 0.0),
        }),
        ..default()
    });
}

fn update_line(
    time: Res<Time>,
    solver: Query<&Solver>,
    mut line: Query<&mut LineStrip>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let executor_data = solver.single();
    let executor = executor_data.solver.lock()
        .expect("Cannot lock solver");
    
    let (_, mesh) = meshes.iter_mut().next().unwrap();

    let x = line.single_mut().points.iter().map(|p| p.x).collect::<Vec<_>>();
    let y = line.single_mut().points.iter().map(|p| p.y).collect::<Vec<_>>();

    // line.single_mut().points.iter_mut().for_each(|p| p.y = (p.y * 2.0 * PI + time.delta_seconds()).sin());

    let job = SplineJob {
        samples: 100,
        x,
        y,
        eps: executor_data.eps,
        max: executor_data.max,
        mode: executor_data.mode,
    };

    let points = executor.solve_spline(job);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, points);

    
}

/// A list of points that will have a line drawn between each consecutive points
#[derive(Debug, Clone, Component)]
pub struct LineStrip {
    pub points: Vec<Vec3>,
}

impl From<LineStrip> for Mesh {
    fn from(line: LineStrip) -> Self {
        Mesh::new(PrimitiveTopology::LineStrip)
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, line.points)
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct LineMaterial {
    #[uniform(0)]
    pub color: Color,
}

impl Material for LineMaterial {
    fn fragment_shader() -> ShaderRef {
        "line_material.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayout,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Set the polygon mode to line
        descriptor.primitive.polygon_mode = PolygonMode::Line;
        Ok(())
    }
}
