//! Product acceptance using real files, the production engine, and the same jobs as the UI.
use aurora_engine::{AdjustmentSet, ExportFormat, ExportJob, FileEngine, ImageEngine, JobStatus, MetadataPolicy, ProcessRecipe};
use std::{fs, path::PathBuf, time::{Duration, Instant}};

struct Fixture(PathBuf);
impl Drop for Fixture { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }

#[test]
fn real_library_edit_close_reopen_export_preserves_original() {
    let dir = std::env::temp_dir().join(format!("aurora-desktop-workflow-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    fs::create_dir(&dir).unwrap(); let temp = Fixture(dir);
    let lib = temp.0.join("图库"); fs::create_dir(&lib).unwrap();
    let input = lib.join("渐变.png");
    image::RgbImage::from_fn(80, 48, |x,y| image::Rgb([(20+x) as u8, (20+y) as u8, 90])).save(&input).unwrap();
    let original = fs::read(&input).unwrap();
    let neutral = lib.join("参考.png"); fs::write(&neutral, &original).unwrap();
    {
        let engine = FileEngine::open(&lib).unwrap();
        let photo = engine.images().into_iter().find(|p| p.file_name == "渐变.png").unwrap();
        let mut edits = AdjustmentSet::default(); edits.exposure.exposure = 0.75;
        let queued = engine.save_adjustments_queued(&[(photo.id, edits)]);
        assert_eq!(queued.accepted(),1);
        assert!(engine.flush_pending_saves(Duration::from_secs(10)).unwrap().is_complete());
    }
    let engine = FileEngine::open(&lib).unwrap(); let images=engine.images();
    let photo = images.iter().find(|p| p.file_name == "渐变.png").unwrap();
    let reference = images.iter().find(|p| p.file_name == "参考.png").unwrap();
    assert_eq!(engine.load_adjustments(photo.id).exposure.exposure, 0.75);
    let out = temp.0.join("成片"); fs::create_dir(&out).unwrap();
    for (job, format, depth) in [(1, ExportFormat::Jpeg, 8), (2, ExportFormat::Png, 8), (3, ExportFormat::Tiff, 16), (4, ExportFormat::Png, 8)] {
        engine.start_export(ExportJob { job_id:job, image:if job==4 {reference.id} else {photo.id}, recipe:ProcessRecipe {format, bit_depth:depth, destination:Some(out.clone()), include_metadata:MetadataPolicy::StripAll, ..Default::default()} }).unwrap();
    }
    let mut luminance = [0_u64; 2];
    for id in 1..=4 {
        let start=Instant::now();
        loop { match engine.job_status(id) {
            JobStatus::Done{output} => {
                assert_eq!(std::fs::canonicalize(output.parent().unwrap()).unwrap(), std::fs::canonicalize(&out).unwrap());
                let decoded=image::open(&output).unwrap(); assert_eq!((decoded.width(),decoded.height()),(80,48));
                if id==3 { assert!(matches!(decoded.color(),image::ColorType::Rgb16|image::ColorType::Rgba16)); }
                if id==2 || id==4 { luminance[(id==4) as usize] = decoded.to_rgb8().as_raw().iter().map(|&v| v as u64).sum(); }
                break;
            }
            JobStatus::Failed{message} => panic!("job {id}: {message}"),
            _ => { assert!(start.elapsed()<Duration::from_secs(30),"job {id} stalled"); std::thread::sleep(Duration::from_millis(10)); }
        }}
    }
    assert_eq!(fs::read_dir(out).unwrap().count(),4,"no staging debris");
    assert!(luminance[0] > luminance[1], "edited PNG must be brighter than its identical neutral input: {luminance:?}");
    engine.start_export(ExportJob {job_id: 5, image: reference.id, recipe: ProcessRecipe::default()}).unwrap();
    let deadline = Instant::now()+Duration::from_secs(30);
    loop {
        match engine.job_status(5) {
            JobStatus::Done{output} => {
                assert_eq!(output.parent().unwrap().canonicalize().unwrap(), lib.join("Output").canonicalize().unwrap(), "default destination must match the GUI label");
                assert_eq!(image::image_dimensions(output).unwrap(), (80,48)); break;
            }
            JobStatus::Failed{message} => panic!("default jpeg: {message}"),
            _ => { assert!(Instant::now()<deadline); std::thread::sleep(Duration::from_millis(10)); }
        }
    }
    drop(engine);
    assert_eq!(FileEngine::open(&lib).unwrap().images().len(), 2, "reopening must not reimport its own exported JPEG");
    assert_eq!(fs::read(&neutral).unwrap(), original);
    assert_eq!(fs::read(&input).unwrap(),original,"edits must not modify the source");
}
