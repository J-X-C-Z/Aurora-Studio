# B8 body shader and ABI

Status: `ABI-COMPLETE / INFERENCE-BLOCKED`.

Base: `eececfd`; B8 prerequisite HEAD: `f36146d`.

Changed contract:

- `BodyPoseModel` accepts only a static `[1,69,2|3]` keypoint output. `KeypointXyz` and `XyzKeypoint` are both normalized to slot order `0..=68`; no semantic name is assigned to an unproven slot.
- `BodyKeypointSet` requires exactly 69 finite normalized points. `BodyLandmarkFrame::from_pose` enters this ABI before producer conversion; missing or short output is refused, never zero-filled.
- Six-point shaders bind six explicit ordered slots through `BodySixPointShaderInput.bind_ordered`. The input contract is `source_points[6]`, `destination_points[6]`, `degrees[6]`, `amount`, `inverse_aspect`, `valid_point_count`.
- Mesh VBO is 16-byte little-endian `f32x4`: destination XY at byte 0 and source UV at byte 8. EBO is little-endian `u16` triangles. `encode_body_geometry` / `decode_body_geometry` are now the shared ABI used by `BodyGeometryJob`.

Executable CPU routes in `BodyGeometryJob`: `400`, `401`, `404`, `405`, `406`, `407`, `408`, `410`, `415`, `416`, `38556`, `90041`. PF400/PF404 consume the six-point ABI; the others remain explicit mask, mesh, UV, brush, or shift payloads.

Model SHA-256: unavailable. B7's `pose-static.manifest.json` has `pose_model: null`; no authenticated model file or digest exists. No real keypoint was produced, and the ABI tests use explicitly synthetic fixtures only.

Blocker: obtain an authenticated 69-point model artifact plus SHA-256 sidecar and run one real inference. A real run must confirm that the model's output order matches the six shader slot bindings. Target six-point coordinates and non-model mask/mesh/UV payloads are still required by their existing contracts.

Verification: changed Rust modules parse with `rustfmt --emit stdout`; `lib.rs` additionally reported the pre-existing missing `aurora_raw/golden_tests.rs` module during resolver traversal. `git diff --check` found only two pre-existing blank-line-at-EOF warnings. Cargo, GPU, and runtime tests were not run per B8 boundary.
