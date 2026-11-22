use super::{icc_dst::{pcs_xyz_to_encoded, photo_inverse_table}, icc_grid::sample_rgb, icc_source::CameraPcs};

pub fn is_id10_profile(profile: &[u8]) -> bool {
    use super::icc_source::icc_tag;

    let description = icc_tag(profile, b"desc");
    profile.get(16..24) == Some(b"RGB XYZ ")
        && description.is_some()
        && [(b"rXYZ", [44137i32, 18287, -127]), (b"gXYZ", [10857, 44259, 1965]),
            (b"bXYZ", [8195, 2989, 52222])].into_iter().all(|(signature, words)| {
            icc_tag(profile, signature).is_some_and(|tag| {
                tag.get(..8) == Some(b"XYZ \0\0\0\0")
                    && tag.get(8..) == Some(words.map(i32::to_be_bytes).as_flattened())
            })
        })
        && [b"rTRC", b"gTRC", b"bTRC"].into_iter().all(|signature| {
            icc_tag(profile, signature) == Some(b"para\0\0\0\0\0\x03\0\0\0\x02\0\0\0\0\xfd\xfb\0\0\x02\x04\0\0\x08\0\0\0\x02\x08")
        })
}

/// Intent 0 without BPC, for the validated Z6III ProStandard pairings.
/// Pass the original ID14 Photo RGB ICC for that target; None selects fixed ID10 BT.2020.
pub fn working_rgb_table(camera: &CameraPcs, photo_profile: Option<&[u8]>, points: u8) -> Option<Vec<u16>> {
    if !matches!(points, 17 | 33 | 49) {
        return None;
    }
    let photo_inverse = match photo_profile {
        Some(profile) => Some(photo_inverse_table(profile)?),
        None => None,
    };
    let mut table = vec![0; 3 * usize::from(points).pow(3)];
    sample_rgb([points; 3], &mut table, |rgb| {
        pcs_xyz_to_encoded(photo_inverse.as_deref(), camera.to_xyz(rgb))
    });
    Some(table)
}
