use aurora_portrait::cuda_job_upload::{
    JOB_SLICE_ALIGN, JobBufferDesc, JobBufferPlan, JobBufferPurpose, JobBufferRole,
};
use aurora_portrait::Refusal;

fn input<'a>(
    role: JobBufferRole,
    ordinal: u16,
    payload: &'a dyn aurora_portrait::cuda_job_upload::JobPayload,
) -> JobBufferDesc<'a> {
    JobBufferDesc {
        role,
        ordinal,
        purpose: JobBufferPurpose::ReadOnlyInput,
        payload,
    }
}

#[test]
fn one_job_description_contains_all_provider_categories() {
    let face = [[0.25f32, 0.75], [0.5, 0.5]];
    let body = [3u16, 1, 4];
    let repair = [255u8; 16];
    let historical = [0.5f32; 8];
    let plan = JobBufferPlan::new(
        91,
        (1024, 768),
        vec![
            input(JobBufferRole::HistoricalSkin, 0, &historical),
            input(JobBufferRole::RepairPlane, 0, &repair),
            input(JobBufferRole::BodyGeometry, 0, &body),
            input(JobBufferRole::FaceGeometry, 0, &face),
        ],
    )
    .expect("valid job");

    assert_eq!(plan.len(), 4);
    assert_eq!(plan.descriptor(0).unwrap().role, JobBufferRole::FaceGeometry);
    assert!(plan.offset(0).unwrap() % JOB_SLICE_ALIGN == 0);
    for index in 1..plan.len() {
        assert!(plan.offset(index).unwrap() >= plan.offset(index - 1).unwrap());
    }
}

#[test]
fn shape_hit_does_not_imply_content_hit() {
    let first = [1.0f32, 2.0, 3.0];
    let second = [1.0f32, 2.0, 4.0];
    let a = JobBufferPlan::new(
        1,
        (512, 512),
        vec![input(JobBufferRole::FaceGeometry, 0, &first)],
    )
    .unwrap();
    let b = JobBufferPlan::new(
        2,
        (512, 512),
        vec![input(JobBufferRole::FaceGeometry, 0, &second)],
    )
    .unwrap();

    assert_eq!(a.shape_key(), b.shape_key());
    assert_ne!(a.content_key(0), b.content_key(0));
    assert_eq!(b.changed_from(&[a.content_key(0).unwrap()]), vec![true]);
}

#[test]
fn duplicate_role_and_ordinal_are_refused() {
    let repair = [1u8, 2, 3];
    let error = match JobBufferPlan::new(
        1,
        (4, 4),
        vec![
            input(JobBufferRole::RepairPlane, 7, &repair),
            input(JobBufferRole::RepairPlane, 7, &repair),
        ],
    ) {
        Ok(_) => panic!("duplicate payload must be refused"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        Refusal::ValueOutOfRange("duplicate CUDA job payload")
    );
}

#[test]
fn empty_provider_plan_is_refused() {
    let error = match JobBufferPlan::new(1, (4, 4), Vec::new()) {
        Ok(_) => panic!("empty job must be refused"),
        Err(error) => error,
    };
    assert_eq!(error, Refusal::MissingInput("CUDA job payloads"));
}
