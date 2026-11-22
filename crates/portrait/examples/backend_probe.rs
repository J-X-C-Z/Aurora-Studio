use aurora_portrait::catalog;

fn print_catalog() {
    let rows = catalog();
    let mut cpu = 0;
    let mut gpu_planned = 0;
    let mut refused = 0;
    for row in &rows {
        match row.access {
            aurora_portrait::Access::Cpu => cpu += 1,
            aurora_portrait::Access::GpuPlanned => gpu_planned += 1,
            aurora_portrait::Access::Refused => refused += 1,
        }
    }
    let total = rows.len();
    println!("catalog: {total} PF/stage rows, not 112 effects");
    println!("access counts: Cpu={cpu} GpuPlanned={gpu_planned} Refused={refused}");
}

fn main() {
    println!("CPU: library linked and catalog loaded");
    print_catalog();

    #[cfg(target_os = "windows")]
    {
        match aurora_portrait::cuda::Nvidia::open() {
            Ok(_) => println!("CUDA: driver/PTX module init OK"),
            Err(error) => println!("CUDA: init FAILED: {error:?}"),
        }
    }

    #[cfg(target_os = "macos")]
    {
        match aurora_portrait::metal::MetalContext::new() {
            Ok(_) => println!("Metal: framework/device/library init OK"),
            Err(error) => println!("Metal: init FAILED: {error:?}"),
        }
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        println!("GPU: unsupported on this platform");
    }
}
