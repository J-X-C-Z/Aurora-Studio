//! Minimal Vulkan compute wrapper for the probe.
//!
//! Deliberately narrow: Vulkan only (`ash` loading `vulkan-1.dll` directly; the
//! crate contains no DX12 / Metal / GL path and never links wgpu), one compute
//! queue, host-visible coherent memory for every buffer, explicit command
//! recording so submit / dispatch / download counts are counted events rather
//! than inferred from the API shape.
//!
//! Ownership: `Buf`s own their memory and release it when dropped, and
//! `Gpu::drop` releases the device plus the pools, fence and layouts it created.
//! Nothing here is a general-purpose allocator; it is sized for a fixture that
//! runs a few dozen small cases.

use ash::{vk, Device, Entry, Instance};
use std::ffi::CString;

pub struct VkCtx {
    pub entry: Entry,
    pub instance: Instance,
    /// Version reported by the loader itself (Vulkan 1.1+). `None` on a 1.0
    /// loader: the probe reports "unknown" instead of inventing a number.
    pub loader_api: Option<(u32, u32, u32)>,
}

impl VkCtx {
    pub fn new() -> Result<Self, String> {
        let entry = unsafe { Entry::load() }
            .map_err(|e| format!("failed to load Vulkan loader (vulkan-1.dll): {e}"))?;
        let loader_api = match unsafe { entry.try_enumerate_instance_version() } {
            Ok(Some(v)) => Some((
                vk::api_version_major(v),
                vk::api_version_minor(v),
                vk::api_version_patch(v),
            )),
            _ => None,
        };
        let app = CString::new(env!("CARGO_PKG_NAME")).unwrap();
        let engine = CString::new(env!("CARGO_PKG_NAME")).unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            // Highest version this probe understands; a lower loader/driver is
            // still enumerable, it just reports fewer features.
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| format!("vkCreateInstance failed: {e:?}"))?;
        Ok(Self { entry, instance, loader_api })
    }

    pub fn loader_api_str(&self) -> String {
        match self.loader_api {
            Some((maj, min, patch)) => format!("{maj}.{min}.{patch}"),
            None => "unknown".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: vk::PhysicalDeviceType,
    pub api: (u32, u32, u32),
    #[allow(dead_code)]
    pub api_raw: u32,
    pub driver_version: u32,
    pub driver_version_human: String,
    pub is_software: bool,
    pub max_compute_invocations: u32,
    pub max_compute_workgroup: [u32; 3],
    pub max_storage_buffer_range: u32,
    pub max_push_constants: u32,
    pub shader_float16: bool,
    pub shader_float64: bool,
    pub shader_int8: bool,
}

impl DeviceInfo {
    pub fn vendor_hex(&self) -> String {
        format!("0x{:04x}", self.vendor_id)
    }
    pub fn device_hex(&self) -> String {
        format!("0x{:04x}", self.device_id)
    }
    pub fn type_str(&self) -> &'static str {
        match self.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => "discrete_gpu",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated_gpu",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual_gpu",
            vk::PhysicalDeviceType::CPU => "cpu_software",
            _ => "other",
        }
    }
    pub fn api_str(&self) -> String {
        format!("{}.{}.{}", self.api.0, self.api.1, self.api.2)
    }
}

fn cstr_to_string(raw: &[std::os::raw::c_char]) -> String {
    let bytes: Vec<u8> = raw
        .iter()
        .take_while(|c| **c != 0)
        .map(|c| *c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).to_string()
}

fn decode_driver(vendor: u32, v: u32) -> String {
    match vendor {
        0x10de => format!("NVIDIA {}.{}.{}.{}", v >> 22, (v >> 14) & 0xff, (v >> 6) & 0xff, v & 0x3f),
        0x1002 | 0x8086 => format!("{}.{}.{}", v >> 22, (v >> 12) & 0x3ff, v & 0xfff),
        0x5143 => format!("Qualcomm {}.{}.{}", v >> 22, (v >> 12) & 0x3ff, v & 0xfff),
        _ => format!("raw 0x{v:08x}"),
    }
}

/// Enumerate every physical device visible to the Vulkan loader and classify it.
pub fn enumerate(ctx: &VkCtx) -> Result<Vec<(vk::PhysicalDevice, DeviceInfo)>, String> {
    let devices = unsafe { ctx.instance.enumerate_physical_devices() }
        .map_err(|e| format!("vkEnumeratePhysicalDevices failed: {e:?}"))?;
    let mut out = Vec::new();
    for (index, pd) in devices.into_iter().enumerate() {
        let p = unsafe { ctx.instance.get_physical_device_properties(pd) };
        let f = unsafe { ctx.instance.get_physical_device_features(pd) };
        // shaderFloat16 / shaderInt8 live in VkPhysicalDeviceVulkan12Features,
        // which may only be queried from a Vulkan 1.2+ device. On an older
        // device both are reported as unavailable rather than queried illegally.
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        if p.api_version >= vk::make_api_version(0, 1, 2, 0) {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f12);
            unsafe { ctx.instance.get_physical_device_features2(pd, &mut f2) };
        }
        let l = &p.limits;
        let info = DeviceInfo {
            index,
            name: cstr_to_string(&p.device_name),
            vendor_id: p.vendor_id,
            device_id: p.device_id,
            device_type: p.device_type,
            api: (
                vk::api_version_major(p.api_version),
                vk::api_version_minor(p.api_version),
                vk::api_version_patch(p.api_version),
            ),
            api_raw: p.api_version,
            driver_version: p.driver_version,
            driver_version_human: decode_driver(p.vendor_id, p.driver_version),
            // Vulkan software implementations (lavapipe / llvmpipe / SwiftShader)
            // report PHYSICAL_DEVICE_TYPE_CPU. They are never accepted as "real
            // hardware" by this probe.
            is_software: p.device_type == vk::PhysicalDeviceType::CPU,
            max_compute_invocations: l.max_compute_work_group_invocations,
            max_compute_workgroup: l.max_compute_work_group_size,
            max_storage_buffer_range: l.max_storage_buffer_range,
            max_push_constants: l.max_push_constants_size,
            shader_float16: f12.shader_float16 != vk::FALSE,
            shader_float64: f.shader_float64 != vk::FALSE,
            shader_int8: f12.shader_int8 != vk::FALSE,
        };
        out.push((pd, info));
    }
    Ok(out)
}

/// A device buffer with its own memory, freed on drop. The device handle is
/// copied in so `Buf` can free itself; that also means a `Buf` must not outlive
/// the `Gpu` that created it (every call site keeps both in one scope, and the
/// helpers below hand out borrowed `&Buf`s only).
pub struct Buf {
    device: Device,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
}

impl Drop for Buf {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// Cumulative per-device instrumentation, so submit/dispatch/download counts in
/// the report are counted events, never inferred from the API shape.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    pub submits: u64,
    pub dispatches: u64,
    pub downloads: u64,
    pub uploads: u64,
    pub upload_bytes: u64,
    pub download_bytes: u64,
}

pub struct Gpu {
    #[allow(dead_code)]
    pub info: DeviceInfo,
    device: Device,
    queue: vk::Queue,
    queue_family: u32,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    desc_pool: vk::DescriptorPool,
    pub desc_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    // instrumentation
    pub submits: u64,
    pub dispatches: u64,
    pub downloads: u64,
    pub uploads: u64,
    pub upload_bytes: u64,
    pub download_bytes: u64,
}

impl Gpu {
    pub fn open(ctx: &VkCtx, pd: vk::PhysicalDevice, info: DeviceInfo) -> Result<Self, String> {
        let families = unsafe { ctx.instance.get_physical_device_queue_family_properties(pd) };
        let (qfam, _) = families
            .iter()
            .enumerate()
            .find(|(_, f)| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, f)| (i as u32, f))
            .ok_or_else(|| "device exposes no COMPUTE queue family".to_string())?;

        let priorities = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfam)
            .queue_priorities(&priorities)];
        let device = unsafe {
            ctx.instance.create_device(
                pd,
                &vk::DeviceCreateInfo::default().queue_create_infos(&qci),
                None,
            )
        }
        .map_err(|e| format!("vkCreateDevice failed: {e:?}"))?;

        let queue = unsafe { device.get_device_queue(qfam, 0) };
        let mem_props = unsafe { ctx.instance.get_physical_device_memory_properties(pd) };

        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qfam)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| format!("vkCreateCommandPool failed: {e:?}"))?;
        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| format!("vkAllocateCommandBuffers failed: {e:?}"))?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| format!("vkCreateFence failed: {e:?}"))?;

        let sizes = [1u32; 5];
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..5u32)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(sizes[i as usize])
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let desc_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|e| format!("vkCreateDescriptorSetLayout failed: {e:?}"))?;
        let set_layouts = [desc_layout];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )
        }
        .map_err(|e| format!("vkCreatePipelineLayout failed: {e:?}"))?;

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(64)];
        let desc_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(64)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|e| format!("vkCreateDescriptorPool failed: {e:?}"))?;

        Ok(Self {
            info,
            device,
            queue,
            queue_family: qfam,
            mem_props,
            cmd_pool,
            cmd,
            fence,
            desc_pool,
            desc_layout,
            pipeline_layout,
            submits: 0,
            dispatches: 0,
            downloads: 0,
            uploads: 0,
            upload_bytes: 0,
            download_bytes: 0,
        })
    }

    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    pub fn snapshot(&self) -> Counters {
        Counters {
            submits: self.submits,
            dispatches: self.dispatches,
            downloads: self.downloads,
            uploads: self.uploads,
            upload_bytes: self.upload_bytes,
            download_bytes: self.download_bytes,
        }
    }

    pub fn since(&self, base: &Counters) -> Counters {
        Counters {
            submits: self.submits - base.submits,
            dispatches: self.dispatches - base.dispatches,
            downloads: self.downloads - base.downloads,
            uploads: self.uploads - base.uploads,
            upload_bytes: self.upload_bytes - base.upload_bytes,
            download_bytes: self.download_bytes - base.download_bytes,
        }
    }

    fn pick_memory(&self, type_bits: u32, want: vk::MemoryPropertyFlags) -> Result<u32, String> {
        // Prefer DEVICE_LOCAL host-visible memory (ReBAR / UMA iGPU), else plain
        // host-visible coherent. Never fall back to non-visible memory: this probe
        // uses map/unmap instead of staging copies on purpose.
        let mut best: Option<u32> = None;
        for i in 0..self.mem_props.memory_type_count {
            if type_bits & (1 << i) == 0 {
                continue;
            }
            let flags = self.mem_props.memory_types[i as usize].property_flags;
            if !flags.contains(want | vk::MemoryPropertyFlags::HOST_COHERENT) {
                continue;
            }
            if flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
                return Ok(i);
            }
            if best.is_none() {
                best = Some(i);
            }
        }
        best.ok_or_else(|| format!("no memory type with flags {want:?} for bits 0x{type_bits:x}"))
    }

    pub fn create_buf(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<Buf, String> {
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|e| format!("vkCreateBuffer({size} B) failed: {e:?}"))?;
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mt = self.pick_memory(req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE)?;
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mt),
                None,
            )
        }
        .map_err(|e| format!("vkAllocateMemory({} B) failed: {e:?}", req.size))?;
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| format!("vkBindBufferMemory failed: {e:?}"))?;
        Ok(Buf { device: self.device.clone(), buffer, memory, size })
    }

    /// Every host transfer goes through this check, so a wrong element count is
    /// an error instead of an out-of-bounds map.
    fn check_range(what: &str, bytes: usize, size: u64) -> Result<(), String> {
        if bytes as u64 > size {
            return Err(format!("{what} {bytes} B exceeds buffer {size} B"));
        }
        Ok(())
    }

    pub fn upload_u32(&mut self, b: &Buf, data: &[u32]) -> Result<(), String> {
        let bytes = data.len() * 4;
        Self::check_range("upload", bytes, b.size)?;
        unsafe {
            let ptr = self
                .device
                .map_memory(b.memory, 0, b.size, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("vkMapMemory(upload) failed: {e:?}"))?;
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, bytes);
            self.device.unmap_memory(b.memory);
        }
        self.uploads += 1;
        self.upload_bytes += bytes as u64;
        Ok(())
    }

    pub fn download_u32(&mut self, b: &Buf, words: usize) -> Result<Vec<u32>, String> {
        let bytes = words * 4;
        Self::check_range("download", bytes, b.size)?;
        let mut out = vec![0u32; words];
        unsafe {
            let ptr = self
                .device
                .map_memory(b.memory, 0, b.size, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("vkMapMemory(download) failed: {e:?}"))?;
            std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr() as *mut u8, bytes);
            self.device.unmap_memory(b.memory);
        }
        self.downloads += 1;
        self.download_bytes += bytes as u64;
        Ok(out)
    }

    pub fn upload_f32(&mut self, b: &Buf, data: &[f32]) -> Result<(), String> {
        let bytes = data.len() * 4;
        Self::check_range("upload", bytes, b.size)?;
        unsafe {
            let ptr = self
                .device
                .map_memory(b.memory, 0, b.size, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("vkMapMemory(upload f32) failed: {e:?}"))?;
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, bytes);
            self.device.unmap_memory(b.memory);
        }
        self.uploads += 1;
        self.upload_bytes += bytes as u64;
        Ok(())
    }

    pub fn download_f32(&mut self, b: &Buf, count: usize) -> Result<Vec<f32>, String> {
        let bytes = count * 4;
        Self::check_range("download", bytes, b.size)?;
        let mut out = vec![0f32; count];
        unsafe {
            let ptr = self
                .device
                .map_memory(b.memory, 0, b.size, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("vkMapMemory(download f32) failed: {e:?}"))?;
            std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr() as *mut u8, bytes);
            self.device.unmap_memory(b.memory);
        }
        self.downloads += 1;
        self.download_bytes += bytes as u64;
        Ok(out)
    }

    pub fn make_pipeline(
        &self,
        spv: &[u32],
        entry: &str,
    ) -> Result<vk::Pipeline, String> {
        let name = CString::new(entry).unwrap();
        let module = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(spv),
                None,
            )
        }
        .map_err(|e| format!("vkCreateShaderModule({entry}) failed: {e:?}"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&name);
        let ci = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.pipeline_layout)];
        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &ci, None)
        };
        let pipeline = match pipelines {
            Ok(mut p) => p.remove(0),
            Err((_, e)) => {
                unsafe { self.device.destroy_shader_module(module, None) };
                return Err(format!("vkCreateComputePipelines({entry}) failed: {e:?}"));
            }
        };
        unsafe { self.device.destroy_shader_module(module, None) };
        Ok(pipeline)
    }

    /// Pipelines are handed out as raw handles (they are cheap to keep and only
    /// exist for the duration of one device run); this releases them explicitly
    /// instead of waiting for `vkDestroyDevice`.
    pub fn destroy_pipeline(&self, p: vk::Pipeline) {
        unsafe { self.device.destroy_pipeline(p, None) };
    }

    /// Allocate one descriptor set for up to 5 storage buffers (skipped = unused).
    pub fn alloc_set(&self, bufs: &[&Buf]) -> Result<vk::DescriptorSet, String> {
        let layouts = [self.desc_layout];
        let set = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.desc_pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(|e| format!("vkAllocateDescriptorSets failed: {e:?}"))?[0];

        let infos: Vec<vk::DescriptorBufferInfo> = bufs
            .iter()
            .map(|b| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        Ok(set)
    }

    pub fn begin(&self) -> Result<(), String> {
        unsafe {
            self.device
                .reset_command_pool(self.cmd_pool, vk::CommandPoolResetFlags::empty())
                .map_err(|e| format!("vkResetCommandPool failed: {e:?}"))?;
            // Descriptor sets are single-run scratch: the previous command buffer
            // has already been submitted and fenced, and the previous case's
            // buffers are already destroyed, so recycling the pool here cannot
            // invalidate anything still in flight.
            self.device
                .reset_descriptor_pool(self.desc_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| format!("vkResetDescriptorPool failed: {e:?}"))?;
            self.device
                .reset_fences(&[self.fence])
                .map_err(|e| format!("vkResetFences failed: {e:?}"))?;
        }
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(self.cmd, &begin) }
            .map_err(|e| format!("vkBeginCommandBuffer failed: {e:?}"))
    }

    /// Host writes -> compute reads.
    pub fn barrier_host_to_compute(&self) {
        let mb = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &mb,
                &[],
                &[],
            )
        };
    }

    /// Dispatch N writes -> next dispatch reads (the unfused intermediate).
    pub fn barrier_compute_to_compute(&self) {
        let mb = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)];
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &mb,
                &[],
                &[],
            )
        };
    }

    pub fn dispatch(
        &mut self,
        pipeline: vk::Pipeline,
        set: vk::DescriptorSet,
        groups: (u32, u32),
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            self.device.cmd_dispatch(self.cmd, groups.0, groups.1, 1);
        }
        self.dispatches += 1;
    }

    /// End, submit exactly one command buffer, wait for the fence.
    pub fn submit_and_wait(&mut self) -> Result<(), String> {
        unsafe {
            self.device
                .end_command_buffer(self.cmd)
                .map_err(|e| format!("vkEndCommandBuffer failed: {e:?}"))?;
            let cmds = [self.cmd];
            let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
            self.device
                .queue_submit(self.queue, &submit, self.fence)
                .map_err(|e| format!("vkQueueSubmit failed: {e:?}"))?;
            self.device
                .wait_for_fences(&[self.fence], true, 60_000_000_000)
                .map_err(|e| format!("vkWaitForFences failed: {e:?}"))?;
        }
        self.submits += 1;
        Ok(())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            // Quiesce first: every command buffer is already fenced, but a
            // failure between submit and wait would otherwise free live objects.
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.desc_layout, None);
            // Implicitly destroys anything still alive on the device (pipelines
            // on an aborted run); `Buf`s free themselves before this point.
            self.device.destroy_device(None);
        }
    }
}

impl Drop for VkCtx {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
        let _ = &self.entry;
    }
}
