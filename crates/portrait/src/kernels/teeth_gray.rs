//! TeethWhitenV3 `USE_GRAY_FUSE` + SoftLightAlpha candidate.
//!
//! Independent port of the embedded final fragment (`0x968E1CE`, 1039 B,
//! SHA256 `1dbb7f928865948ff1a63d7322b669da014178c15b8c06094ff76d2bcf31c9a0`):
//! `G = dot(src, (0.298912, 0.586611, 0.114478))` multiplies both LUT weights
//! before whiten/bright; `ch = SoftLightAlpha` applies `mix(x, sqrt(x), ch)`
//! after both LUT stages; alpha is preserved.
//!
//! Same `Image`/`Refusal` conventions as `kernels::face_detail`; the LUT reader
//! and 8-bit store thresholds are duplicated here so this file stays standalone.

use crate::{Image, Refusal};

const INV255: f32 = 1.0 / 255.0;
pub const LUT_LEN: usize = 512 * 512 * 3;
pub const GATE: f32 = 0.001;
pub const GRAY_COEFFICIENTS: [f32; 3] = [0.298912, 0.586611, 0.114478];
const TEETH_STORE_THRESHOLDS: [f64; 256] = [
    0.5605493055190891, 0.556803893763572, 0.553058541379869, 0.5493133077397943,
    0.545567836612463, 0.5418230779469013, 0.5380773693323135, 0.5330834276974201,
    0.5293377190828323, 0.5255929604172707, 0.5218472518026829, 0.5181024931371212,
    0.5143567845225334, 0.5106110759079456, 0.5056180842220783, 0.5018733255565166,
    0.5605487674474716, 0.5568040087819099, 0.5530592501163483, 0.5493125915527344,
    0.5455678328871727, 0.5405748412013054, 0.5368281826376915, 0.5330834239721298,
    0.5293386653065681, 0.5255920067429543, 0.5218472480773926, 0.5181024894118309,
    0.514355830848217, 0.5093628391623497, 0.505618080496788, 0.5018733218312263,
    0.5605516135692596, 0.5568049550056458, 0.5530582964420319, 0.549311637878418,
    0.5443186461925507, 0.5405719876289368, 0.5368253290653229, 0.5330862700939178,
    0.529339611530304, 0.5255929529666901, 0.5218462944030762, 0.5168533027172089,
    0.513106644153595, 0.5093637853860855, 0.505620926618576, 0.5018742680549622,
    0.5605506598949432, 0.5568040013313293, 0.551811009645462, 0.5480643510818481,
    0.5443176925182343, 0.5405748337507248, 0.5368281751871109, 0.5330853164196014,
    0.5293386578559875, 0.5255919992923737, 0.5205990076065063, 0.5168523490428925,
    0.5131056904792786, 0.5093666315078735, 0.5056199729442596, 0.5018733143806458,
    0.5605497062206268, 0.5555567145347595, 0.5518100559711456, 0.5480633974075317,
    0.5443167388439178, 0.540570080280304, 0.5368234217166901, 0.533084362745285,
    0.5280913710594177, 0.5243447124958038, 0.5205980539321899, 0.516851395368576,
    0.5131047368049622, 0.5093580782413483, 0.5056190192699432, 0.5018723607063293,
    0.559302419424057, 0.5555557608604431, 0.5518091022968292, 0.5480624437332153,
    0.5443157851696014, 0.5405767261981964, 0.5368300676345825, 0.5318370759487152,
    0.5280904173851013, 0.5243437588214874, 0.5205971002578735, 0.5168504416942596,
    0.5131037831306458, 0.5093647241592407, 0.5056180655956268, 0.5006250739097595,
    0.5593014657497406, 0.5555548071861267, 0.5518081486225128, 0.5480614900588989,
    0.5443224310874939, 0.5393294394016266, 0.5355827808380127, 0.5318361222743988,
    0.5280894637107849, 0.524342805147171, 0.5205961465835571, 0.5168570876121521,
    0.5131104290485382, 0.5081174373626709, 0.504370778799057, 0.5006241202354431,
    0.5593005120754242, 0.5555538535118103, 0.5518147945404053, 0.5480681359767914,
    0.5430675446987152, 0.5393284857273102, 0.5355818271636963, 0.5318351686000824,
    0.5280885100364685, 0.5243418514728546, 0.5206027925014496, 0.5168561339378357,
    0.5118631422519684, 0.5081164836883545, 0.5043698251247406, 0.5630462169647217,
    0.5592919588088989, 0.5555528998374939, 0.5505675077438354, 0.5468132495880127,
    0.5430741906166077, 0.5393199324607849, 0.5355808734893799, 0.5318266153335571,
    0.5280875563621521, 0.5243332982063293, 0.5193479061126709, 0.5156088471412659,
    0.5118545889854431, 0.5081155300140381, 0.5043612718582153, 0.5630452632904053,
    0.5592910051345825, 0.5543056130409241, 0.550566554069519, 0.5468122959136963,
    0.5430732369422913, 0.5393189787864685, 0.5355799198150635, 0.5318256616592407,
    0.5280866026878357, 0.5231012105941772, 0.5193469524383545, 0.5156078934669495,
    0.5118536353111267, 0.5081145763397217, 0.5043603181838989, 0.5617979764938354,
    0.5580589175224304, 0.5543046593666077, 0.5505656003952026, 0.5468113422393799,
    0.5430722832679749, 0.5393180251121521, 0.5355789661407471, 0.5305935740470886,
    0.5268393158912659, 0.5231002569198608, 0.5193459987640381, 0.5156069397926331,
    0.5118526816368103, 0.5081136226654053, 0.5043593645095825, 0.561797022819519,
    0.558057963848114, 0.5543037056922913, 0.5505646467208862, 0.5468103885650635,
    0.5430713295936584, 0.5393170714378357, 0.5343316793441772, 0.5305926203727722,
    0.5268383622169495, 0.5230993032455444, 0.5193450450897217, 0.5156059861183167,
    0.5118517279624939, 0.5068663358688354, 0.5031272768974304, 0.5617960691452026,
    0.5580570101737976, 0.5543027520179749, 0.5505636930465698, 0.5468094348907471,
    0.5418240427970886, 0.5380697846412659, 0.5343307256698608, 0.5305916666984558,
    0.5268374085426331, 0.523098349571228, 0.5193440914154053, 0.5156050324440002,
    0.5106044411659241, 0.506865382194519, 0.503126323223114, 0.5617951154708862,
    0.5580560564994812, 0.5543017983436584, 0.54931640625, 0.5455621480941772,
    0.5418230891227722, 0.5380688309669495, 0.5343297719955444, 0.5305907130241394,
    0.5268364548683167, 0.5230973958969116, 0.5180968046188354, 0.5143577456474304,
    0.5106034874916077, 0.5068644285202026, 0.5031253695487976, 0.5617941617965698,
    0.5580551028251648, 0.5530545115470886, 0.5493154525756836, 0.5455611944198608,
    0.5418221354484558, 0.5380678772926331, 0.534328818321228, 0.530589759349823,
    0.5268355011940002, 0.5218501091003418, 0.518095850944519, 0.514356791973114,
    0.5106025338172913, 0.5068634748458862, 0.5031244158744812, 0.560546875,
    0.556807816028595, 0.5530535578727722, 0.5493144989013672, 0.5455602407455444,
    0.5418211817741394, 0.5380669236183167, 0.5343278646469116, 0.5293424725532532,
    0.5255882143974304, 0.5218491554260254, 0.5180948972702026, 0.5143558382987976,
    0.5106015801429749, 0.5068625211715698, 0.5031234622001648, 255.0,
];
fn store_unorm8(value: f32) -> u8 {
    let v = value.max(0.0).min(1.0) as f64 * 255.0;
    let floor = v.floor();
    let index = floor.clamp(0.0, 255.0) as usize;
    let fraction = v - floor;
    let mut result = index as i32;
    if fraction >= TEETH_STORE_THRESHOLDS[index] {
        result += 1;
    }
    result.clamp(0, 255) as u8
}

fn sample_lut(table: &[u8], r: f32, g: f32, b: f32, degree: f32) -> [u8; 3] {
    let blue = b * 63.0;
    let fb = blue.floor();
    let cb = blue.ceil();
    let q1y = (fb / 8.0).floor();
    let q1x = fb - q1y * 8.0;
    let q2y = (cb / 8.0).floor();
    let q2x = cb - q2y * 8.0;
    let fetch = |quad_x: f32, quad_y: f32| -> [f32; 3] {
        let sx = quad_x * 0.125 + 0.0009765625 + 0.123046875 * r;
        let sy = quad_y * 0.125 + 0.0009765625 + 0.123046875 * g;
        let xs = sx * 512.0 - 0.5;
        let ys = sy * 512.0 - 0.5;
        let xi = xs.floor() as i32;
        let yi = ys.floor() as i32;
        let ax = ((((xs - xi as f32) * 256.0) + 0.5).floor()) / 256.0;
        let ay = ((((ys - yi as f32) * 256.0) + 0.5).floor()) / 256.0;
        let x0 = xi.clamp(0, 511) as usize;
        let x1 = (xi + 1).clamp(0, 511) as usize;
        let y0 = yi.clamp(0, 511) as usize;
        let y1 = (yi + 1).clamp(0, 511) as usize;
        let mut out = [0.0f32; 3];
        for channel in 0..3 {
            let top = table[(y0 * 512 + x0) * 3 + channel] as f32 / 255.0 * (1.0 - ax)
                + table[(y0 * 512 + x1) * 3 + channel] as f32 / 255.0 * ax;
            let bottom = table[(y1 * 512 + x0) * 3 + channel] as f32 / 255.0 * (1.0 - ax)
                + table[(y1 * 512 + x1) * 3 + channel] as f32 / 255.0 * ax;
            out[channel] = top * (1.0 - ay) + bottom * ay;
        }
        out
    };
    let c1 = fetch(q1x, q1y);
    let c2 = fetch(q2x, q2y);
    let fraction = blue - fb;
    let mut out = [0u8; 3];
    for channel in 0..3 {
        let sampled = c1[channel] * (1.0 - fraction) + c2[channel] * fraction;
        let src = [r, g, b][channel];
        out[channel] = store_unorm8(src * (1.0 - degree) + sampled * degree);
    }
    out
}

/// `mask_rgba` is the combine mask (`RGBA=(dz*G, G*ea, 0, 1)`), one RGBA byte
/// per source pixel. `degree` is the final `en` uniform (1.0 for this effect).
pub fn teeth_whitening_gray(
    source: &Image,
    mask_rgba: &[u8],
    whiten_lut: &[u8],
    bright_lut: &[u8],
    degree: f32,
    use_gray_fuse: bool,
    softlight_alpha: f32,
) -> Result<Image, Refusal> {
    let pixels = (source.width as usize)
        .checked_mul(source.height as usize)
        .ok_or(Refusal::ShapeMismatch("source"))?;
    if source.pixels.len() != pixels * 4 {
        return Err(Refusal::ShapeMismatch("source"));
    }
    if mask_rgba.len() != pixels * 4 {
        return Err(Refusal::ShapeMismatch("mask_rgba"));
    }
    if whiten_lut.len() != LUT_LEN {
        return Err(Refusal::ShapeMismatch("whiten_lut"));
    }
    if bright_lut.len() != LUT_LEN {
        return Err(Refusal::ShapeMismatch("bright_lut"));
    }
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("degree"));
    }
    if !softlight_alpha.is_finite() || softlight_alpha < 0.0 {
        return Err(Refusal::ValueOutOfRange("softlight_alpha"));
    }

    let mut output = source.clone();
    for p in 0..pixels {
        let base = p * 4;
        let mut r = source.pixels[base] as f32 * INV255;
        let mut g = source.pixels[base + 1] as f32 * INV255;
        let mut b = source.pixels[base + 2] as f32 * INV255;

        let mut whiten = mask_rgba[base] as f32 * INV255 * degree;
        let mut bright = mask_rgba[base + 1] as f32 * INV255 * degree;
        if use_gray_fuse {
            let gray = r * GRAY_COEFFICIENTS[0]
                + g * GRAY_COEFFICIENTS[1]
                + b * GRAY_COEFFICIENTS[2];
            whiten *= gray;
            bright *= gray;
        }

        if whiten > GATE {
            let sampled = sample_lut(whiten_lut, r, g, b, whiten);
            r = sampled[0] as f32 * INV255;
            g = sampled[1] as f32 * INV255;
            b = sampled[2] as f32 * INV255;
        }
        if bright > GATE {
            let sampled = sample_lut(bright_lut, r, g, b, bright);
            r = sampled[0] as f32 * INV255;
            g = sampled[1] as f32 * INV255;
            b = sampled[2] as f32 * INV255;
        }
        if softlight_alpha > 0.0 {
            r = r * (1.0 - softlight_alpha) + r.max(0.0).sqrt() * softlight_alpha;
            g = g * (1.0 - softlight_alpha) + g.max(0.0).sqrt() * softlight_alpha;
            b = b * (1.0 - softlight_alpha) + b.max(0.0).sqrt() * softlight_alpha;
        }

        output.pixels[base] = store_unorm8(r);
        output.pixels[base + 1] = store_unorm8(g);
        output.pixels[base + 2] = store_unorm8(b);
    }
    Ok(output)
}
#[cfg(test)]
mod tests {
    use super::*;

    fn lut() -> Vec<u8> {
        // Identity ramp so the LUT itself does not change the pixel.
        let mut table = Vec::with_capacity(LUT_LEN);
        for y in 0..512usize {
            for x in 0..512usize {
                table.push((x % 256) as u8);
                table.push((y % 256) as u8);
                table.push(((x + y) % 256) as u8);
            }
        }
        table
    }

    #[test]
    fn refusals_are_typed() {
        let source = Image::solid(1, 1, [10, 20, 30, 40]);
        let mask = [0u8; 4];
        assert_eq!(teeth_whitening_gray(&source, &mask[..3], &lut(), &lut(), 1.0, false, 0.0),
            Err(Refusal::ShapeMismatch("mask_rgba")));
        assert_eq!(teeth_whitening_gray(&source, &mask, &[], &lut(), 1.0, false, 0.0),
            Err(Refusal::ShapeMismatch("whiten_lut")));
        assert_eq!(teeth_whitening_gray(&source, &mask, &lut(), &lut(), 1.5, false, 0.0),
            Err(Refusal::ValueOutOfRange("degree")));
        assert_eq!(teeth_whitening_gray(&source, &mask, &lut(), &lut(), 1.0, false, -1.0),
            Err(Refusal::ValueOutOfRange("softlight_alpha")));
    }

    #[test]
    fn zero_weight_keeps_the_pixel_and_alpha() {
        let source = Image::solid(1, 1, [10, 20, 30, 40]);
        let out = teeth_whitening_gray(&source, &[0, 0, 0, 255], &lut(), &lut(), 1.0, false, 0.0)
            .unwrap();
        assert_eq!(out, source);
        // SoftLightAlpha runs unconditionally when > 0, even with zero weights.
        let out = teeth_whitening_gray(&source, &[0, 0, 0, 255], &lut(), &lut(), 1.0, true, 1.0)
            .unwrap();
        assert_ne!(out, source, "sqrt soft light applies with ch > 0");
        assert_eq!(out.pixel(0, 0)[3], source.pixel(0, 0)[3]);
    }
}
