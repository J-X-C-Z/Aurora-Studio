//! Background-task bookkeeping split out of [`super::AppState`].
//!
//! This is a plain data holder: export job ordering and the two "busy" flags
//! are the only state it owns. It deliberately has no event bus of its own;
//! the UI still observes `AppState`.

use aurora_engine::JobStatus;

use super::JobView;

/// Export jobs plus the AI-mask and selection-analysis busy flags.
///
/// Field semantics are unchanged from the former `AppState` fields:
/// `next_job` starts at 1, ids are consumed even when the engine rejects the
/// export, and `jobs` keeps submission order.
#[derive(Clone, Debug)]
pub struct TaskCenter {
    /// Export jobs in submission order.
    pub jobs: Vec<JobView>,
    next_job: u64,
    /// An AI mask computation is running.
    pub ai_busy: bool,
    /// A selection analysis (eyes, rotation, Match Look) is running.
    pub analysing: bool,
}

impl Default for TaskCenter {
    fn default() -> Self {
        Self { jobs: Vec::new(), next_job: 1, ai_busy: false, analysing: false }
    }
}

impl TaskCenter {
    /// Reserves the next job id without recording a job.
    ///
    /// [`super::AppState::process`] needs the id for the engine's `ExportJob`
    /// before it knows whether the export was accepted; a rejected export
    /// still consumes the id, as before.
    pub fn reserve_job_id(&mut self) -> u64 {
        let id = self.next_job;
        self.next_job += 1;
        id
    }

    /// Records an accepted export job, keeping submission order.
    pub fn push_job(&mut self, job: JobView) {
        self.jobs.push(job);
    }

    /// Reserves an id and records a queued job in one step.
    pub fn start_job(&mut self, name: String, recipe: String) -> u64 {
        let id = self.reserve_job_id();
        self.jobs.push(JobView { id, name, recipe, status: JobStatus::Queued });
        id
    }

    /// The job with `id`, for status updates and cancellation.
    pub fn job_mut(&mut self, id: u64) -> Option<&mut JobView> {
        self.jobs.iter_mut().find(|job| job.id == id)
    }

    /// Sets a job's status, if it is still listed.
    pub fn finish_job(&mut self, id: u64, status: JobStatus) -> bool {
        match self.job_mut(id) {
            Some(job) => {
                job.status = status;
                true
            }
            None => false,
        }
    }
}
