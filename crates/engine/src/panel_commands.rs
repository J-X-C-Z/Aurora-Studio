//! Thin engine wrappers for the Levin panels that are still placeholders.
//!
//! These are additive commands. They do not change the existing merge, ICC,
//! print or tether paths; they give the frontend stable names and result types
//! while keeping unsupported parts explicit.

use anyhow::{ensure, Context, Result};
use serde::Serialize;

use crate::print::{self, PageSpec, PrintTemplate};
use crate::{ImageId, RgbaImage};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CaptureDevice {
    pub id: String,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PortraitCapability {
    pub pf: u32,
    pub key: String,
    pub label: String,
    pub stage: &'static str,
    pub required_inputs: Vec<String>,
    pub engine_bridge: bool,
    pub automatic: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrintRequest {
    pub images: Vec<ImageId>,
    /// Paper name accepted by [`print::paper_mm`], e.g. `A4` or `Letter`.
    pub paper: String,
    pub landscape: bool,
    pub rows: u32,
    pub columns: u32,
    pub margin_mm: f32,
    pub spacing_mm: f32,
    pub fill: bool,
    pub captions: bool,
    pub dpi: u32,
}

impl Default for PrintRequest {
    fn default() -> Self {
        Self {
            images: Vec::new(),
            paper: "A4".into(),
            landscape: false,
            rows: 2,
            columns: 2,
            margin_mm: 12.0,
            spacing_mm: 6.0,
            fill: false,
            captions: true,
            dpi: 300,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrintLayout {
    pub paper_mm: (f32, f32),
    pub page_px: (u32, u32),
    pub cell_mm: (f32, f32),
    pub per_page: usize,
    pub pages: usize,
}

#[derive(Clone, Debug)]
pub struct PrintPreview {
    pub page: usize,
    pub image: RgbaImage,
    pub layout: PrintLayout,
}

pub(crate) fn print_template(request: &PrintRequest) -> Result<(PrintTemplate, PrintLayout)> {
    ensure!(!request.images.is_empty(), "print needs at least one image");
    ensure!(request.dpi >= 72 && request.dpi <= 1200, "print dpi must be 72..=1200");
    ensure!(request.rows > 0 && request.columns > 0, "print layout needs at least one cell");
    ensure!(request.margin_mm >= 0.0 && request.spacing_mm >= 0.0, "print margins and spacing must be non-negative");
    let (mut width_mm, mut height_mm) = print::paper_mm(&request.paper)
        .with_context(|| format!("unknown paper size {}", request.paper))?;
    if request.landscape {
        std::mem::swap(&mut width_mm, &mut height_mm);
    }
    let rows = request.rows as f32;
    let columns = request.columns as f32;
    let cell_width_mm = (width_mm - 2.0 * request.margin_mm - (columns - 1.0) * request.spacing_mm) / columns;
    let cell_height_mm = (height_mm - 2.0 * request.margin_mm - (rows - 1.0) * request.spacing_mm) / rows;
    ensure!(cell_width_mm > 0.0 && cell_height_mm > 0.0, "paper margins/spacing leave no printable cell");
    let per_page = request.rows as usize * request.columns as usize;
    let pages = request.images.len().div_ceil(per_page);
    let spec = PageSpec { width_mm, height_mm, margin_mm: 0.0, dpi: request.dpi };
    let page_px = spec.px()?;
    let template = PrintTemplate {
        margins: print::TemplateMargins {
            left_mm: request.margin_mm,
            top_mm: request.margin_mm,
            right_mm: request.margin_mm,
            bottom_mm: request.margin_mm,
        },
        layout: print::TemplateLayout {
            rows: request.rows,
            columns: request.columns,
            row_gap_mm: request.spacing_mm,
            column_gap_mm: request.spacing_mm,
            cell_width_mm,
            cell_height_mm,
            ..Default::default()
        },
        image_options: print::TemplateImageOptions {
            zoom_to_fill: request.fill,
            ..Default::default()
        },
        ..Default::default()
    };
    Ok((
        template,
        PrintLayout { paper_mm: (width_mm, height_mm), page_px, cell_mm: (cell_width_mm, cell_height_mm), per_page, pages },
    ))
}

pub(crate) fn dynamic_image<E: crate::ImageEngine + ?Sized>(engine: &E, id: ImageId) -> Result<image::DynamicImage> {
    let adjustments = engine.load_adjustments(id);
    let thumb = engine.thumbnail(id, &adjustments, 4096)?;
    let width = thumb.width;
    let height = thumb.height;
    let pixels = thumb.pixels.as_ref().clone();
    let image = image::RgbaImage::from_raw(width, height, pixels)
        .context("engine returned a thumbnail of the wrong size")?;
    Ok(image::DynamicImage::ImageRgba8(image))
}

pub(crate) fn print_images<E: crate::ImageEngine + ?Sized>(engine: &E, request: &PrintRequest) -> Result<Vec<image::DynamicImage>> {
    request.images.iter().map(|id| dynamic_image(engine, *id)).collect()
}

pub(crate) fn print_page<E: crate::ImageEngine + ?Sized>(engine: &E, request: &PrintRequest, page: usize) -> Result<(image::RgbImage, PageSpec)> {
    let (template, _) = print_template(request)?;
    let images = print_images(engine, request)?;
    print::compose_template_page(&template, &images, page)
}

fn rgb_to_rgba(rgb: image::RgbImage) -> RgbaImage {
    let (width, height) = rgb.dimensions();
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for pixel in rgb.pixels() {
        pixels.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
    }
    RgbaImage::new(width, height, pixels)
}

/// The complete portrait catalogue with the engine's own bridge state.
///
/// `engine_bridge` is derived from the dispatcher: it is true exactly when
/// `portrait_stage` reads `pc:<pf>` and hands the frame to a recovered kernel
/// (or a typed refusal naming the missing same-job input). It is not a parity or
/// acceptance claim, and `automatic` stays false because no row has real-photo
/// acceptance for the desktop default packaging.
pub(crate) fn portrait_capabilities() -> Vec<PortraitCapability> {
    aurora_portrait::catalog()
        .into_iter()
        .map(|meta| {
            let stage = match meta.access {
                aurora_portrait::Access::Cpu => "cpu_stage",
                aurora_portrait::Access::GpuPlanned => "parameters_only",
                aurora_portrait::Access::Refused => "unavailable",
            };
            PortraitCapability {
                pf: meta.pf,
                key: if meta.key.is_empty() {
                    format!("pf{}", meta.pf)
                } else {
                    meta.key.to_string()
                },
                label: meta.label.to_string(),
                stage,
                required_inputs: meta.inputs.iter().map(|input| (*input).to_string()).collect(),
                engine_bridge: stage == "cpu_stage"
                    && crate::portrait_stage::is_bridged_effect(meta.pf),
                automatic: false,
            }
        })
        .collect()
}

pub(crate) fn print_preview<E: crate::ImageEngine + ?Sized>(engine: &E, request: &PrintRequest, page: usize) -> Result<PrintPreview> {
    let (_, layout) = print_template(request)?;
    let (canvas, _) = print_page(engine, request, page)?;
    Ok(PrintPreview { page, image: rgb_to_rgba(canvas), layout })
}

pub(crate) fn print_to_pdf<E: crate::ImageEngine + ?Sized>(engine: &E, request: &PrintRequest, output: &std::path::Path) -> Result<print::RenderedPage> {
    let (canvas, spec) = print_page(engine, request, 0)?;
    print::write_image_pdf(&image::DynamicImage::ImageRgb8(canvas), output, &spec)
}

pub(crate) fn print_to_printer<E: crate::ImageEngine + ?Sized>(
    engine: &E,
    request: &PrintRequest,
    printer: Option<&str>,
) -> std::result::Result<(), print::PrintError> {
    let (_, layout) = print_template(request).map_err(|error| print::PrintError::Invalid(error.to_string()))?;
    if layout.pages > 1 {
        return Err(print::PrintError::Unsupported { page_count: layout.pages });
    }
    let (canvas, spec) = print_page(engine, request, 0).map_err(|error| print::PrintError::Render(error.to_string()))?;
    print::print_rgb_page(&canvas, &spec, printer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageEngine;

    struct Fixture(std::path::PathBuf);

    impl Fixture {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "aurora-panel-{name}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn image(&self, name: &str) -> std::path::PathBuf {
            let path = self.0.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            image::RgbaImage::from_pixel(32, 24, image::Rgba([20, 40, 60, 255])).save(&path).unwrap();
            path
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn engine_with_two_images(name: &str) -> (Fixture, crate::FileEngine) {
        let fixture = Fixture::new(name);
        fixture.image("a.png");
        fixture.image("b.png");
        let engine = crate::FileEngine::open(&fixture.0).unwrap();
        (fixture, engine)
    }

    #[test]
    fn print_layout_computes_cells_and_pages() {
        let engine = crate::MockEngine::new();
        let mut request = PrintRequest::default();
        request.images = vec![ImageId(1), ImageId(2), ImageId(3), ImageId(4), ImageId(5)];
        let layout = engine.print_layout(&request).unwrap();
        assert_eq!(layout.per_page, 4);
        assert_eq!(layout.pages, 2);
        assert!(layout.cell_mm.0 > 0.0 && layout.cell_mm.1 > 0.0);
        assert!(layout.page_px.0 > 0 && layout.page_px.1 > 0);
        assert!(layout.page_px.1 > layout.page_px.0, "default A4 is portrait");
        let pixel_ratio = layout.page_px.0 as f32 / layout.page_px.1 as f32;
        let paper_ratio = layout.paper_mm.0 / layout.paper_mm.1;
        assert!((pixel_ratio - paper_ratio).abs() < 0.01);
        request.landscape = true;
        let landscape = engine.print_layout(&request).unwrap();
        assert!(landscape.page_px.0 > landscape.page_px.1);
    }

    #[test]
    fn print_to_printer_rejects_multiple_pages_with_the_page_count() {
        let engine = crate::MockEngine::new();
        let mut request = PrintRequest::default();
        request.images = vec![ImageId(1), ImageId(2), ImageId(3), ImageId(4), ImageId(5)];
        let error = engine.print_to_printer(&request, None).unwrap_err();
        assert_eq!(error, print::PrintError::Unsupported { page_count: 2 });
    }

    #[test]
    fn print_to_printer_reports_no_printer_for_an_unknown_name() {
        let engine = crate::MockEngine::new();
        let mut request = PrintRequest::default();
        request.images = vec![ImageId(1)];
        let error = engine
            .print_to_printer(&request, Some("Aurora No Such Printer"))
            .unwrap_err();
        assert_eq!(error, print::PrintError::NoPrinter);
    }

    #[test]
    fn print_layout_rejects_unknown_paper_and_bad_margins() {
        let engine = crate::MockEngine::new();
        let mut request = PrintRequest::default();
        request.images = vec![ImageId(1)];
        request.paper = "No Such Paper".into();
        assert!(engine.print_layout(&request).is_err());
        request.paper = "A4".into();
        request.margin_mm = 200.0;
        assert!(engine.print_layout(&request).is_err());
    }

    #[test]
    fn print_preview_and_pdf_write_a_page() {
        let (fixture, engine) = engine_with_two_images("print");
        let request = PrintRequest { images: engine.images().iter().map(|image| image.id).collect(), ..Default::default() };
        let preview = engine.print_preview(&request, 0).unwrap();
        assert_eq!(preview.layout.per_page, 4);
        assert_eq!(preview.image.width, preview.layout.page_px.0);
        let output = fixture.0.join("print.pdf");
        let page = engine.print_to_pdf(&request, &output).unwrap();
        assert_eq!(page.output, output);
        assert!(std::fs::read(&output).unwrap().starts_with(b"%PDF"));
    }

    #[test]
    fn print_preview_rejects_an_out_of_range_page() {
        let (_fixture, engine) = engine_with_two_images("print-page");
        let request = PrintRequest { images: vec![engine.images()[0].id], ..Default::default() };
        assert!(engine.print_preview(&request, 2).is_err());
    }

    #[test]
    fn print_to_pdf_rejects_an_empty_selection() {
        let (_fixture, engine) = engine_with_two_images("print-empty");
        assert!(engine.print_to_pdf(&PrintRequest::default(), &_fixture.0.join("empty.pdf")).is_err());
    }

    #[test]
    fn capture_devices_is_empty_on_a_mock_engine() {
        let engine = crate::MockEngine::new();
        assert!(engine.capture_devices().is_empty());
    }

    #[test]
    fn capture_devices_on_file_engine_does_not_panic() {
        let (_fixture, engine) = engine_with_two_images("capture");
        let _ = engine.capture_devices();
    }

    #[test]
    fn capture_start_reports_the_missing_session() {
        let engine = crate::MockEngine::new();
        assert!(engine.capture_start().is_err());
    }

    #[test]
    fn capture_start_on_file_engine_reports_the_missing_session() {
        let (_fixture, engine) = engine_with_two_images("capture-start");
        assert!(engine.capture_start().is_err());
    }

    #[test]
    fn capture_stop_is_idempotent_and_arrived_is_none() {
        let engine = crate::MockEngine::new();
        assert!(engine.capture_stop().is_ok());
        assert!(engine.capture_arrived().is_none());
    }

    #[test]
    fn capture_stop_and_arrived_on_file_engine_are_safe() {
        let (_fixture, engine) = engine_with_two_images("capture-stop");
        assert!(engine.capture_stop().is_ok());
        assert!(engine.capture_arrived().is_none());
    }

    #[test]
    fn portrait_capabilities_are_registered_on_file_engine() {
        let (_fixture, engine) = engine_with_two_images("portrait");
        let capabilities = engine.portrait_capabilities();
        assert!(!capabilities.is_empty());
        assert!(capabilities.iter().any(|capability| capability.pf == 90005));
        assert!(capabilities.iter().all(|capability| !capability.automatic));
    }

    #[test]
    fn portrait_capabilities_report_the_derived_bridge_state() {
        let capabilities = portrait_capabilities();
        let catalogue = aurora_portrait::catalog();
        assert_eq!(capabilities.len(), catalogue.len());
        let bridged = crate::portrait_stage::bridged_effects();
        let claimed = capabilities
            .iter()
            .filter(|capability| capability.engine_bridge)
            .map(|capability| capability.pf)
            .collect::<Vec<_>>();
        assert_eq!(claimed, bridged);
        assert!(capabilities.windows(2).all(|pair| pair[0].pf < pair[1].pf));
        for capability in &capabilities {
            assert!(!capability.automatic, "automatic needs real-photo acceptance");
            assert_eq!(
                capability.engine_bridge,
                capability.stage == "cpu_stage"
                    && crate::portrait_stage::is_bridged_effect(capability.pf),
            );
        }
        // Rows that only declare parameters must never claim a bridge.
        for capability in capabilities.iter().filter(|row| row.stage != "cpu_stage") {
            assert!(!capability.engine_bridge, "PF{}", capability.pf);
        }
    }

    #[test]
    fn portrait_capabilities_are_empty_on_a_mock_engine() {
        assert!(crate::MockEngine::new().portrait_capabilities().is_empty());
    }
}
