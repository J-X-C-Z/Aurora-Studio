//! Single-page PDF output for the print dialog.
//!
//! The page is composited at the requested DPI on a white sheet, so the PDF
//! MediaBox and the embedded image both reflect the selected paper.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use image::{DynamicImage, ExtendedColorType, ImageEncoder, RgbImage, RgbaImage};

/// Why a system-printer request could not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrintError {
    /// No system printer is installed or the named printer is unavailable.
    NoPrinter,
    /// The current printer path only renders one page.
    Unsupported { page_count: usize },
    /// The print request itself is invalid.
    Invalid(String),
    /// Rendering the page failed before it reached the printer.
    Render(String),
    /// A Win32 call failed.
    Win32 { context: &'static str, code: u32 },
}

impl std::fmt::Display for PrintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPrinter => write!(f, "no system printer is available"),
            Self::Unsupported { page_count } => {
                write!(f, "printing {page_count} pages is not supported by this build")
            }
            Self::Invalid(message) => write!(f, "invalid print request: {message}"),
            Self::Render(message) => write!(f, "render failed: {message}"),
            Self::Win32 { context, code } => write!(f, "{context} failed (Win32 {code})"),
        }
    }
}

impl std::error::Error for PrintError {}

/// Names visible to the Windows spooler. Non-Windows builds return an empty
/// list, which lets the frontend show the same `NoPrinter` state.
#[cfg(windows)]
pub fn printer_names() -> Vec<String> {
    win::printer_names()
}

#[cfg(not(windows))]
pub fn printer_names() -> Vec<String> {
    Vec::new()
}

/// Prints one already-composited page. Callers must reject multi-page requests
/// before this function; it prints exactly the bitmap it receives.
#[cfg(windows)]
pub fn print_rgb_page(page: &RgbImage, _spec: &PageSpec, printer: Option<&str>) -> std::result::Result<(), PrintError> {
    win::print_page(page, printer)
}

#[cfg(not(windows))]
pub fn print_rgb_page(_page: &RgbImage, _spec: &PageSpec, _printer: Option<&str>) -> std::result::Result<(), PrintError> {
    Err(PrintError::NoPrinter)
}

#[cfg(windows)]
mod win {
    use super::{PrintError, RgbImage};
    use std::ffi::c_void;
    use std::ptr::{null, null_mut};

    const PRINTER_ENUM_LOCAL: u32 = 2;
    const PRINTER_ENUM_CONNECTIONS: u32 = 4;
    const HORZRES: i32 = 8;
    const VERTRES: i32 = 10;
    const HALFTONE: i32 = 4;
    const SRCCOPY: u32 = 0x00CC_0020;

    #[repr(C)]
    struct PrinterInfo4W {
        printer_name: *mut u16,
        server_name: *mut u16,
        attributes: u32,
    }

    #[repr(C)]
    struct DocInfoW {
        cb_size: i32,
        doc_name: *const u16,
        output: *const u16,
        datatype: *const u16,
        fw_type: u32,
    }

    #[repr(C)]
    struct BitmapInfoHeader {
        size: u32,
        width: i32,
        height: i32,
        planes: u16,
        bit_count: u16,
        compression: u32,
        size_image: u32,
        x_pels_per_meter: i32,
        y_pels_per_meter: i32,
        clr_used: u32,
        clr_important: u32,
    }

    #[repr(C)]
    struct RgbQuad {
        blue: u8,
        green: u8,
        red: u8,
        reserved: u8,
    }

    #[repr(C)]
    struct BitmapInfo {
        header: BitmapInfoHeader,
        colors: [RgbQuad; 1],
    }

    #[link(name = "winspool")]
    unsafe extern "system" {
        fn EnumPrintersW(
            flags: u32,
            name: *const u16,
            level: u32,
            buffer: *mut u8,
            buffer_size: u32,
            needed: *mut u32,
            returned: *mut u32,
        ) -> i32;
        fn GetDefaultPrinterW(buffer: *mut u16, size: *mut u32) -> i32;
    }

    #[link(name = "gdi32")]
    unsafe extern "system" {
        fn CreateDCW(
            driver: *const u16,
            device: *const u16,
            port: *const u16,
            dev_mode: *const c_void,
        ) -> *mut c_void;
        fn DeleteDC(hdc: *mut c_void) -> i32;
        fn StartDocW(hdc: *mut c_void, info: *const DocInfoW) -> i32;
        fn EndDoc(hdc: *mut c_void) -> i32;
        fn StartPage(hdc: *mut c_void) -> i32;
        fn EndPage(hdc: *mut c_void) -> i32;
        fn SetStretchBltMode(hdc: *mut c_void, mode: i32) -> i32;
        fn GetDeviceCaps(hdc: *mut c_void, index: i32) -> i32;
        fn StretchDIBits(
            hdc: *mut c_void,
            x_dest: i32,
            y_dest: i32,
            w_dest: i32,
            h_dest: i32,
            x_src: i32,
            y_src: i32,
            w_src: i32,
            h_src: i32,
            bits: *const c_void,
            info: *const BitmapInfo,
            usage: u32,
            rop: u32,
        ) -> i32;
    }

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(Some(0)).collect()
    }

    unsafe fn wide_string(ptr: *const u16) -> String {
        if ptr.is_null() {
            return String::new();
        }
        let mut len = 0;
        while *ptr.add(len) != 0 {
            len += 1;
        }
        String::from_utf16_lossy(std::slice::from_raw_parts(ptr, len))
    }

    fn last_error(context: &'static str) -> PrintError {
        PrintError::Win32 {
            context,
            code: std::io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(0) as u32,
        }
    }

    pub(super) fn printer_names() -> Vec<String> {
        unsafe {
            let flags = PRINTER_ENUM_LOCAL | PRINTER_ENUM_CONNECTIONS;
            let mut needed = 0u32;
            let mut returned = 0u32;
            let _ = EnumPrintersW(flags, null(), 4, null_mut(), 0, &mut needed, &mut returned);
            if needed == 0 {
                return Vec::new();
            }
            let words = (needed as usize).div_ceil(std::mem::size_of::<usize>());
            let mut buffer = vec![0usize; words];
            if EnumPrintersW(
                flags,
                null(),
                4,
                buffer.as_mut_ptr() as *mut u8,
                needed,
                &mut needed,
                &mut returned,
            ) == 0
            {
                return Vec::new();
            }
            let infos = buffer.as_ptr() as *const PrinterInfo4W;
            (0..returned as usize)
                .filter_map(|index| {
                    let name = wide_string((*infos.add(index)).printer_name);
                    (!name.is_empty()).then_some(name)
                })
                .collect()
        }
    }

    fn default_printer() -> Option<String> {
        unsafe {
            let mut size = 0u32;
            let _ = GetDefaultPrinterW(null_mut(), &mut size);
            if size == 0 {
                return None;
            }
            let mut buffer = vec![0u16; size as usize];
            if GetDefaultPrinterW(buffer.as_mut_ptr(), &mut size) == 0 {
                return None;
            }
            let len = buffer.iter().position(|value| *value == 0).unwrap_or(buffer.len());
            Some(String::from_utf16_lossy(&buffer[..len]))
        }
    }

    pub(super) fn print_page(page: &RgbImage, printer: Option<&str>) -> Result<(), PrintError> {
        let names = printer_names();
        if names.is_empty() {
            return Err(PrintError::NoPrinter);
        }
        let name = match printer.map(str::trim).filter(|name| !name.is_empty()) {
            Some(name) => name.to_owned(),
            None => default_printer().ok_or(PrintError::NoPrinter)?,
        };
        if !names.iter().any(|candidate| candidate.eq_ignore_ascii_case(&name)) {
            return Err(PrintError::NoPrinter);
        }

        let (width, height) = page.dimensions();
        if width == 0 || height == 0 {
            return Err(PrintError::Invalid("the print page is empty".into()));
        }
        let driver = wide("WINSPOOL");
        let device = wide(&name);
        unsafe {
            let hdc = CreateDCW(driver.as_ptr(), device.as_ptr(), null(), null());
            if hdc.is_null() {
                return Err(PrintError::NoPrinter);
            }
            let dest_width = GetDeviceCaps(hdc, HORZRES);
            let dest_height = GetDeviceCaps(hdc, VERTRES);
            if dest_width <= 0 || dest_height <= 0 {
                let error = last_error("GetDeviceCaps");
                DeleteDC(hdc);
                return Err(error);
            }

            let doc_name = wide("Aurora Studio print");
            let info = DocInfoW {
                cb_size: std::mem::size_of::<DocInfoW>() as i32,
                doc_name: doc_name.as_ptr(),
                output: null(),
                datatype: null(),
                fw_type: 0,
            };
            if StartDocW(hdc, &info) <= 0 {
                let error = last_error("StartDoc");
                DeleteDC(hdc);
                return Err(error);
            }
            if StartPage(hdc) <= 0 {
                let error = last_error("StartPage");
                EndDoc(hdc);
                DeleteDC(hdc);
                return Err(error);
            }

            SetStretchBltMode(hdc, HALFTONE);
            let mut bits = Vec::with_capacity((width * height * 4) as usize);
            for pixel in page.pixels() {
                bits.extend_from_slice(&[pixel[2], pixel[1], pixel[0], 255]);
            }
            let bitmap = BitmapInfo {
                header: BitmapInfoHeader {
                    size: std::mem::size_of::<BitmapInfoHeader>() as u32,
                    width: width as i32,
                    height: -(height as i32),
                    planes: 1,
                    bit_count: 32,
                    compression: 0,
                    size_image: 0,
                    x_pels_per_meter: 0,
                    y_pels_per_meter: 0,
                    clr_used: 0,
                    clr_important: 0,
                },
                colors: [RgbQuad { blue: 0, green: 0, red: 0, reserved: 0 }],
            };
            let drawn = StretchDIBits(
                hdc,
                0,
                0,
                dest_width,
                dest_height,
                0,
                0,
                width as i32,
                height as i32,
                bits.as_ptr() as *const c_void,
                &bitmap,
                0,
                SRCCOPY,
            );
            if drawn <= 0 {
                let error = last_error("StretchDIBits");
                let _ = EndPage(hdc);
                EndDoc(hdc);
                DeleteDC(hdc);
                return Err(error);
            }
            if EndPage(hdc) <= 0 {
                let error = last_error("EndPage");
                EndDoc(hdc);
                DeleteDC(hdc);
                return Err(error);
            }
            if EndDoc(hdc) <= 0 {
                let error = last_error("EndDoc");
                DeleteDC(hdc);
                return Err(error);
            }
            DeleteDC(hdc);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PageSpec {
    pub width_mm: f32,
    pub height_mm: f32,
    pub margin_mm: f32,
    pub dpi: u32,
}

impl PageSpec {
    pub fn a4() -> Self {
        Self { width_mm: 210.0, height_mm: 297.0, margin_mm: 12.0, dpi: 300 }
    }

    pub fn px(&self) -> Result<(u32, u32)> {
        if self.width_mm <= 0.0 || self.height_mm <= 0.0 {
            bail!("paper size must be positive");
        }
        if self.margin_mm < 0.0 || self.margin_mm * 2.0 >= self.width_mm.min(self.height_mm) {
            bail!("margins leave no printable area");
        }
        if !(72..=1200).contains(&self.dpi) {
            bail!("DPI must be 72..=1200");
        }
        Ok((
            (self.width_mm / 25.4 * self.dpi as f32).round() as u32,
            (self.height_mm / 25.4 * self.dpi as f32).round() as u32,
        ))
    }

    pub fn points(&self) -> (f32, f32) {
        (self.width_mm * 72.0 / 25.4, self.height_mm * 72.0 / 25.4)
    }

    pub fn content_px(&self) -> Result<(u32, u32)> {
        let (w, h) = self.px()?;
        let margin = (self.margin_mm / 25.4 * self.dpi as f32).round() as u32;
        Ok((w.saturating_sub(margin * 2).max(1), h.saturating_sub(margin * 2).max(1)))
    }
}

#[derive(Clone, Debug)]
pub struct PageLayout {
    pub page_px: (u32, u32),
    pub content_px: (u32, u32),
    pub image_px: (u32, u32),
    pub image_origin: (u32, u32),
}

#[derive(Clone, Debug)]
pub struct RenderedPage {
    pub output: PathBuf,
    pub layout: PageLayout,
    pub page_pt: (f32, f32),
}

/// A Capture One `.coprinttemplate`, read-only. Geometry is stored in PDF
/// points in the file and converted to millimetres here. The template does not
/// contain paper size, DPI, printer or watermark settings.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrintTemplate {
    pub version: Option<u32>,
    pub name: Option<String>,
    pub margins: TemplateMargins,
    pub layout: TemplateLayout,
    pub image_options: TemplateImageOptions,
    pub annotation: TemplateAnnotation,
    /// Elements this parser does not model.
    pub skipped_fields: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TemplateMargins {
    pub left_mm: f32,
    pub top_mm: f32,
    pub right_mm: f32,
    pub bottom_mm: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TemplateLayout {
    pub resizing_rule: u32,
    pub rows: u32,
    pub columns: u32,
    pub row_gap_mm: f32,
    pub column_gap_mm: f32,
    pub cell_height_mm: f32,
    pub cell_width_mm: f32,
    pub cell_border_mm: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TemplateImageOptions {
    pub zoom_to_fill: bool,
    pub rotate_to_fill: bool,
    pub repeat_one_image_per_page: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TemplateAnnotation {
    pub kind: Option<String>,
    pub font_name: Option<String>,
    pub font_size: f32,
    pub foreground_color: Option<String>,
}

const PT_TO_MM: f32 = 25.4 / 72.0;

/// Reads the fields the bundled templates use. Unknown elements are skipped and
/// counted, so a newer template still loads. Not wired into the PDF writer yet.
pub fn parse_print_template(text: &str) -> Result<PrintTemplate> {
    let doc = roxmltree::Document::parse(text).context("parse .coprinttemplate")?;
    let root = doc.root_element();
    if root.tag_name().name() != "PrintTemplate" {
        bail!("not a .coprinttemplate: <{}>", root.tag_name().name());
    }

    let mut out = PrintTemplate::default();
    for node in root.descendants().filter(|n| n.is_element()) {
        if node == root {
            continue;
        }
        let text = node.text().unwrap_or("").trim();
        match node.tag_name().name() {
            "Margin" | "Layout" | "ImageOptions" | "Annotation" | "Template" => {}
            "Version" => out.version = Some(whole(text, "Version")?),
            "Left" => out.margins.left_mm = millimetres(text, "Margin/Left")?,
            "Top" => out.margins.top_mm = millimetres(text, "Margin/Top")?,
            "Right" => out.margins.right_mm = millimetres(text, "Margin/Right")?,
            "Bottom" => out.margins.bottom_mm = millimetres(text, "Margin/Bottom")?,
            "ResizingRule" => out.layout.resizing_rule = whole(text, "Layout/ResizingRule")?,
            "Rows" => out.layout.rows = whole(text, "Layout/Rows")?,
            "Columns" => out.layout.columns = whole(text, "Layout/Columns")?,
            "RowsSpacing" => out.layout.row_gap_mm = millimetres(text, "Layout/RowsSpacing")?,
            "ColumnsSpacing" => out.layout.column_gap_mm = millimetres(text, "Layout/ColumnsSpacing")?,
            "CellHeight" => out.layout.cell_height_mm = millimetres(text, "Layout/CellHeight")?,
            "CellWidth" => out.layout.cell_width_mm = millimetres(text, "Layout/CellWidth")?,
            "CellBorder" => out.layout.cell_border_mm = millimetres(text, "Layout/CellBorder")?,
            "ZoomToFill" => out.image_options.zoom_to_fill = boolean(text, "ImageOptions/ZoomToFill")?,
            "RotateToFill" => out.image_options.rotate_to_fill = boolean(text, "ImageOptions/RotateToFill")?,
            "RepeatOneImagePerPage" => out.image_options.repeat_one_image_per_page = boolean(text, "ImageOptions/RepeatOneImagePerPage")?,
            "Type" => out.annotation.kind = Some(text.to_owned()),
            "FontName" => out.annotation.font_name = Some(text.to_owned()),
            "FontSize" => out.annotation.font_size = number(text, "Annotation/FontSize")?,
            "ForegroundColor" => out.annotation.foreground_color = (!text.is_empty()).then(|| text.to_owned()),
            "Name" => out.name = Some(text.to_owned()),
            other => {
                out.skipped_fields += 1;
                log::debug!("print template: skipped <{other}>");
            }
        }
    }
    Ok(out)
}

fn number(text: &str, field: &str) -> Result<f32> {
    text.parse::<f32>().with_context(|| format!("{field}: not a number: {text:?}"))
}

fn whole(text: &str, field: &str) -> Result<u32> {
    text.parse::<u32>().with_context(|| format!("{field}: not an integer: {text:?}"))
}

fn millimetres(text: &str, field: &str) -> Result<f32> {
    Ok(number(text, field)? * PT_TO_MM)
}

fn boolean(text: &str, field: &str) -> Result<bool> {
    match text.to_ascii_lowercase().as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => bail!("{field}: not a boolean: {text:?}"),
    }
}

/// Renders one page of a template grid to PDF bytes. The page size is
/// reconstructed from the template's margins and cells; the file has no paper
/// size or DPI, so 300 dpi is assumed. Annotation, watermark and cell borders
/// are not drawn.
pub fn render_template_page(template: &PrintTemplate, images: &[DynamicImage], page: usize) -> Result<Vec<u8>> {
    let (canvas, spec) = compose_template_page(template, images, page)?;
    let jpeg = encode_jpeg(&canvas)?;
    Ok(build_pdf(&jpeg, (canvas.width(), canvas.height()), spec.points()))
}

pub fn compose_template_page(template: &PrintTemplate, images: &[DynamicImage], page: usize) -> Result<(RgbImage, PageSpec)> {
    let spec = template_page_spec(template)?;
    let (page_w, page_h) = spec.px()?;
    let rows = template.layout.rows as usize;
    let cols = template.layout.columns as usize;
    let per_page = rows * cols;
    if images.is_empty() {
        bail!("no images to place");
    }
    let pages = images.len().div_ceil(per_page);
    if page >= pages {
        bail!("page {page} out of range ({pages} pages)");
    }

    let scale = spec.dpi as f32 / 25.4;
    let cell_w = (template.layout.cell_width_mm * scale).round() as u32;
    let cell_h = (template.layout.cell_height_mm * scale).round() as u32;
    if cell_w == 0 || cell_h == 0 {
        bail!("template cell is smaller than one pixel");
    }
    let mut canvas = RgbImage::from_pixel(page_w, page_h, image::Rgb([255, 255, 255]));
    for (i, image) in images[page * per_page..].iter().take(per_page).enumerate() {
        let (row, col) = (i / cols, i % cols);
        let cell_x = (template.margins.left_mm + col as f32 * (template.layout.cell_width_mm + template.layout.column_gap_mm)) * scale;
        let cell_y = (template.margins.top_mm + row as f32 * (template.layout.cell_height_mm + template.layout.row_gap_mm)) * scale;
        let rgba = composite_on_white(&image.to_rgba8());
        let (w, h) = fit(rgba.width(), rgba.height(), cell_w, cell_h);
        let resized = image::imageops::resize(&rgba, w, h, image::imageops::FilterType::Lanczos3);
        let x = cell_x.round() as i64 + ((cell_w - w) / 2) as i64;
        let y = cell_y.round() as i64 + ((cell_h - h) / 2) as i64;
        image::imageops::replace(&mut canvas, &resized, x, y);
    }
    Ok((canvas, spec))
}

fn template_page_spec(template: &PrintTemplate) -> Result<PageSpec> {
    let (rows, cols) = (template.layout.rows, template.layout.columns);
    if rows == 0 || cols == 0 {
        bail!("template has no cells");
    }
    if template.layout.cell_width_mm <= 0.0 || template.layout.cell_height_mm <= 0.0 {
        bail!("template cell size must be positive");
    }
    let grid_w = cols as f32 * template.layout.cell_width_mm + (cols - 1) as f32 * template.layout.column_gap_mm;
    let grid_h = rows as f32 * template.layout.cell_height_mm + (rows - 1) as f32 * template.layout.row_gap_mm;
    let width_mm = template.margins.left_mm + grid_w + template.margins.right_mm;
    let height_mm = template.margins.top_mm + grid_h + template.margins.bottom_mm;
    Ok(PageSpec { width_mm, height_mm, margin_mm: 0.0, dpi: 300 })
}

pub fn paper_mm(name: &str) -> Option<(f32, f32)> {
    match name.trim().to_ascii_lowercase().as_str() {
        "a4" => Some((210.0, 297.0)),
        "a3" => Some((297.0, 420.0)),
        "a5" => Some((148.0, 210.0)),
        "letter" => Some((215.9, 279.4)),
        "legal" => Some((215.9, 355.6)),
        "4x6" | "4 x 6 in" => Some((101.6, 152.4)),
        "5x7" | "5 x 7 in" => Some((127.0, 177.8)),
        "8x10" | "8 x 10 in" => Some((203.2, 254.0)),
        _ => None,
    }
}

pub fn write_file_pdf(input: &Path, output: &Path, spec: &PageSpec) -> Result<RenderedPage> {
    let image = image::open(input).with_context(|| format!("decode {}", input.display()))?;
    write_image_pdf(&image, output, spec)
}

pub fn write_image_pdf(image: &DynamicImage, output: &Path, spec: &PageSpec) -> Result<RenderedPage> {
    let (canvas, layout) = compose_page(image, spec)?;
    let jpeg = encode_jpeg(&canvas)?;
    let page_pt = spec.points();
    let pdf = build_pdf(&jpeg, layout.page_px, page_pt);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(output, pdf).with_context(|| format!("write {}", output.display()))?;
    Ok(RenderedPage { output: output.to_path_buf(), layout, page_pt })
}

pub fn compose_page(image: &DynamicImage, spec: &PageSpec) -> Result<(RgbImage, PageLayout)> {
    let (page_w, page_h) = spec.px()?;
    let (content_w, content_h) = spec.content_px()?;
    let rgba = composite_on_white(&image.to_rgba8());
    let (image_w, image_h) = fit(rgba.width(), rgba.height(), content_w, content_h);
    let resized = image::imageops::resize(&rgba, image_w, image_h, image::imageops::FilterType::Lanczos3);
    let margin = (spec.margin_mm / 25.4 * spec.dpi as f32).round() as u32;
    let x = margin + (content_w - image_w) / 2;
    let y = margin + (content_h - image_h) / 2;
    let mut canvas = RgbImage::from_pixel(page_w, page_h, image::Rgb([255, 255, 255]));
    image::imageops::replace(&mut canvas, &resized, x as i64, y as i64);
    Ok((
        canvas,
        PageLayout {
            page_px: (page_w, page_h),
            content_px: (content_w, content_h),
            image_px: (image_w, image_h),
            image_origin: (x, y),
        },
    ))
}

fn fit(width: u32, height: u32, box_w: u32, box_h: u32) -> (u32, u32) {
    let scale = (box_w as f64 / width.max(1) as f64)
        .min(box_h as f64 / height.max(1) as f64);
    (
        ((width as f64 * scale).round() as u32).clamp(1, box_w),
        ((height as f64 * scale).round() as u32).clamp(1, box_h),
    )
}

fn composite_on_white(image: &RgbaImage) -> RgbImage {
    RgbImage::from_fn(image.width(), image.height(), |x, y| {
        let p = image.get_pixel(x, y).0;
        let a = p[3] as u32;
        image::Rgb([
            ((p[0] as u32 * a + 255 * (255 - a) + 127) / 255) as u8,
            ((p[1] as u32 * a + 255 * (255 - a) + 127) / 255) as u8,
            ((p[2] as u32 * a + 255 * (255 - a) + 127) / 255) as u8,
        ])
    })
}

fn encode_jpeg(canvas: &RgbImage) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 95)
        .write_image(canvas.as_raw(), canvas.width(), canvas.height(), ExtendedColorType::Rgb8)
        .context("encode page JPEG")?;
    Ok(out)
}

fn build_pdf(jpeg: &[u8], page_px: (u32, u32), page_pt: (f32, f32)) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");
    let mut offsets = Vec::new();

    let start_object = |out: &mut Vec<u8>, offsets: &mut Vec<usize>| {
        offsets.push(out.len());
    };

    start_object(&mut out, &mut offsets);
    out.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    start_object(&mut out, &mut offsets);
    out.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    start_object(&mut out, &mut offsets);
    out.extend_from_slice(
        format!(
            "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {:.4} {:.4}] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
            page_pt.0, page_pt.1
        )
        .as_bytes(),
    );

    start_object(&mut out, &mut offsets);
    out.extend_from_slice(
        format!(
            "4 0 obj\n<< /Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {} >>\nstream\n",
            page_px.0, page_px.1, jpeg.len()
        )
        .as_bytes(),
    );
    out.extend_from_slice(jpeg);
    out.extend_from_slice(b"\nendstream\nendobj\n");

    let content = format!("q\n{:.4} 0 0 {:.4} 0 0 cm\n/Im0 Do\nQ\n", page_pt.0, page_pt.1);
    start_object(&mut out, &mut offsets);
    out.extend_from_slice(
        format!("5 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes(),
    );
    out.extend_from_slice(content.as_bytes());
    out.extend_from_slice(b"endstream\nendobj\n");

    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", offsets.len() + 1).as_bytes());
    for offset in &offsets {
        out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len() + 1,
            xref
        )
        .as_bytes(),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<PrintTemplate>
  <Version>1</Version>
  <Margin><Left>72</Left><Top>36</Top><Right>18</Right><Bottom>0</Bottom></Margin>
  <Layout>
    <ResizingRule>2</ResizingRule><Rows>4</Rows><Columns>2</Columns>
    <RowsSpacing>28.3464567</RowsSpacing><ColumnsSpacing>56.6929134</ColumnsSpacing>
    <CellHeight>255.1181102</CellHeight><CellWidth>368.503937</CellWidth>
    <CellBorder>1.5</CellBorder>
    <FutureField>ignored</FutureField>
  </Layout>
  <ImageOptions><ZoomToFill>true</ZoomToFill><RotateToFill>false</RotateToFill><RepeatOneImagePerPage>true</RepeatOneImagePerPage></ImageOptions>
  <Annotation><Type>Filename</Type><FontName>Tahoma</FontName><FontSize>8</FontSize><ForegroundColor>#ffffff</ForegroundColor></Annotation>
  <Template><Name>Sample</Name></Template>
</PrintTemplate>"#;

    #[test]
    fn print_template_reads_geometry_and_counts_unknown_fields() {
        let t = parse_print_template(SAMPLE).unwrap();
        assert_eq!(t.version, Some(1));
        assert_eq!(t.name.as_deref(), Some("Sample"));
        assert!((t.margins.left_mm - 25.4).abs() < 1e-4);
        assert!((t.margins.top_mm - 12.7).abs() < 1e-4);
        assert!((t.margins.right_mm - 6.35).abs() < 1e-4);
        assert_eq!(t.margins.bottom_mm, 0.0);
        assert_eq!((t.layout.rows, t.layout.columns), (4, 2));
        assert!((t.layout.row_gap_mm - 10.0).abs() < 1e-3);
        assert!((t.layout.column_gap_mm - 20.0).abs() < 1e-3);
        assert!((t.layout.cell_height_mm - 90.0).abs() < 1e-3);
        assert!((t.layout.cell_border_mm - 1.5 * PT_TO_MM).abs() < 1e-4);
        assert!(t.image_options.zoom_to_fill && !t.image_options.rotate_to_fill && t.image_options.repeat_one_image_per_page);
        assert_eq!(t.annotation.kind.as_deref(), Some("Filename"));
        assert_eq!(t.annotation.foreground_color.as_deref(), Some("#ffffff"));
        assert_eq!(t.skipped_fields, 1);
    }

    #[test]
    fn print_template_rejects_broken_xml() {
        assert!(parse_print_template("<PrintTemplate><Margin>").is_err());
        assert!(parse_print_template("").is_err());
        assert!(parse_print_template("<NotATemplate/>").is_err());
        assert!(parse_print_template("<PrintTemplate><Rows>four</Rows></PrintTemplate>").is_err());
    }

    #[test]
    #[ignore = "needs the local Capture One PrintTemplates folder"]
    fn installed_print_templates_all_parse() {
        let root = PathBuf::from(std::env::var_os("ProgramFiles").expect("ProgramFiles"))
            .join("Capture One").join("Capture One").join("PrintTemplates");
        let mut files = Vec::new();
        collect_templates(&root, &mut files);
        files.sort();
        assert_eq!(files.len(), 70, "expected 70 installed templates");
        eprintln!("parsed {} templates under {}", files.len(), root.display());

        for path in &files {
            let text = std::fs::read_to_string(path).unwrap();
            let t = parse_print_template(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert_eq!(t.version, Some(1), "{}", path.display());
            assert_eq!(t.skipped_fields, 0, "{}", path.display());
        }

        let get = |rel: &str| files.iter().find(|p| p.ends_with(rel)).unwrap_or_else(|| panic!("missing {rel}"));
        let a = parse_print_template(&std::fs::read_to_string(get("A3 Metric\\A3  9x13 cm (8).coprinttemplate")).unwrap()).unwrap();
        assert_eq!((a.layout.rows, a.layout.columns), (4, 2));
        assert!((a.margins.left_mm - 11.6769).abs() < 1e-3);
        assert!((a.margins.top_mm - 13.9252).abs() < 1e-3);

        let b = parse_print_template(&std::fs::read_to_string(get("A3 Metric\\A3 18x24 cm (2).coprinttemplate")).unwrap()).unwrap();
        assert_eq!((b.layout.rows, b.layout.columns), (2, 1));
        assert!((b.margins.top_mm - 20.0).abs() < 1e-3);
        assert!((b.layout.cell_height_mm - 180.0).abs() < 1e-3);

        let c = parse_print_template(&std::fs::read_to_string(get("Full Page.coprinttemplate")).unwrap()).unwrap();
        assert_eq!((c.layout.rows, c.layout.columns), (1, 1));
        assert!((c.layout.cell_width_mm - 296.926).abs() < 0.01);
        assert!((c.layout.cell_height_mm - 420.116).abs() < 0.01);
    }

    fn collect_templates(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                collect_templates(&path, out);
            } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("coprinttemplate")) {
                out.push(path);
            }
        }
    }

    fn solid(color: [u8; 3]) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(8, 8, image::Rgb(color)))
    }

    fn grid_template() -> PrintTemplate {
        PrintTemplate {
            version: Some(1),
            name: Some("2x2".into()),
            layout: TemplateLayout { rows: 2, columns: 2, cell_width_mm: 50.0, cell_height_mm: 50.0, ..Default::default() },
            ..Default::default()
        }
    }

    #[test]
    fn template_page_places_four_images_in_a_two_by_two_grid() {
        let template = grid_template();
        let images = [solid([255, 0, 0]), solid([0, 255, 0]), solid([0, 0, 255]), solid([255, 255, 0])];
        let (canvas, spec) = compose_template_page(&template, &images, 0).unwrap();
        assert_eq!(spec.px().unwrap(), (canvas.width(), canvas.height()));
        let (w, h) = (canvas.width(), canvas.height());
        for (x, y) in [(w / 4, h / 4), (3 * w / 4, h / 4), (w / 4, 3 * h / 4), (3 * w / 4, 3 * h / 4)] {
            assert_ne!(canvas.get_pixel(x, y).0, [255, 255, 255], "cell at {x},{y} is white");
        }

        let pdf = render_template_page(&template, &images, 0).unwrap();
        assert!(pdf.starts_with(b"%PDF-"));
        let marker = b"/Type /Page ";
        assert_eq!(pdf.windows(marker.len()).filter(|w| w.starts_with(marker)).count(), 1);
    }

    #[test]
    fn template_page_rejects_an_empty_image_list() {
        assert!(render_template_page(&grid_template(), &[], 0).is_err());
    }

    #[test]
    fn a4_at_300_dpi_is_2480_by_3508() {
        assert_eq!(PageSpec::a4().px().unwrap(), (2480, 3508));
    }

    #[test]
    fn fit_keeps_aspect_and_stays_in_the_content_box() {
        assert_eq!(fit(1200, 800, 1000, 1000), (1000, 667));
        assert_eq!(fit(100, 100, 300, 200), (200, 200));
    }

    #[test]
    fn a4_page_is_a_single_pdf_page_with_the_right_media_box() {
        let dir = std::env::temp_dir().join(format!("portrait-print-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("page.pdf");
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(40, 30, image::Rgb([120, 90, 60])));
        let page = write_image_pdf(&image, &out, &PageSpec::a4()).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        assert!(bytes.starts_with(b"%PDF-"));
        assert!(bytes.windows(9).any(|w| w == b"/MediaBox"));
        assert!((page.page_pt.0 - 595.28).abs() < 0.2 && (page.page_pt.1 - 841.89).abs() < 0.2);
        assert!(page.layout.image_px.0 > 0 && page.layout.image_px.1 > 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn printer_enumeration_returns_only_named_printers() {
        for name in printer_names() {
            assert!(!name.trim().is_empty(), "EnumPrinters returned an empty name");
        }
    }
}
