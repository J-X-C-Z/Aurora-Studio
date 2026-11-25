#!/usr/bin/env bash
# Build Aurora Studio as an ARM64 macOS application bundle.
#
# This is a LOCAL DEVELOPMENT build:
#   * bundle id  : local.aurora-studio.dev (placeholder, not a registered identity)
#   * signature  : none (scripts/package-macos.sh applies an ad-hoc signature)
#   * notarized  : no
#
# Reproducible inputs: --locked, an explicit aarch64-apple-darwin target, and
# an explicit MACOSX_DEPLOYMENT_TARGET (see below). Rust does not promise
# bit-identical binaries, so "reproducible" here means the same source, lock
# file, target, deployment target and toolchain produce an equivalent bundle.
set -euo pipefail
umask 022

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$ROOT"

APP_NAME="Aurora Studio"
BIN_NAME="aurora-studio"
PACKAGE_NAME="portrait-app"
TARGET="aarch64-apple-darwin"
HOST_ARCH="$(uname -m)"

# --- Deployment target -------------------------------------------------------
# gpui-pre-apple 0.3.11 compiles its Metal shader library with the hardcoded
# flag -mmacosx-version-min=12.0 (its build.rs), so 12.0 is the lowest system
# version this dependency set can honestly claim. gpui-pre-macos runtime-gates
# newer APIs (URL-scheme registration on 12.0, window restoration on 15.0,
# titlebar transparency on 15.3). Raising the value is allowed; lowering it
# below 12.0 is not, because the prebuilt metallib would not match.
# Rationale recorded in docs/macos-adaptation/50-app-audit.md.
DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"
export MACOSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET"

if [ "$(uname -s)" != "Darwin" ]; then
    fail "this script assembles a macOS .app and must run on macOS (host: $(uname -s))"
fi

case "$DEPLOYMENT_TARGET" in
    12.*|13.*|14.*|15.*|16.*|17.*|18.*|19.*|20.*|2[1-9].*) ;;
    *) fail "MACOSX_DEPLOYMENT_TARGET=$DEPLOYMENT_TARGET is below the 12.0 floor required by gpui-pre-apple's Metal library" ;;
esac

# --- Version ----------------------------------------------------------------
VERSION="$(sed -n '/^\[workspace\.package\]/,/^\[/s/^version[[:space:]]*=[[:space:]]*"\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -n 1)"
[ -n "$VERSION" ] || fail "cannot read [workspace.package] version from $ROOT/Cargo.toml"
printf '%s' "$VERSION" | grep -Eq '^[0-9]+(\.[0-9]+){1,3}$' || fail "workspace version '$VERSION' is not a dotted numeric version"

BUNDLE_ID="local.aurora-studio.dev"
DIST_DIR="$ROOT/dist"
APP_DIR="$DIST_DIR/$APP_NAME.app"
CONTENTS="$APP_DIR/Contents"

# --- Toolchain checks -------------------------------------------------------
command -v cargo >/dev/null 2>&1 || fail "cargo is not on PATH"
command -v xcrun >/dev/null 2>&1 || fail "xcrun is not on PATH; install Xcode or Command Line Tools"
command -v plutil >/dev/null 2>&1 || fail "plutil is not on PATH"
command -v file >/dev/null 2>&1 || fail "file is not on PATH"
command -v lipo >/dev/null 2>&1 || fail "lipo is not on PATH"

# gpui-pre-apple compiles src/shaders.metal during the build. The Metal
# compiler ships with Xcode, not with the plain Command Line Tools on every
# release, so fail early with an actionable message.
xcrun --sdk macosx --find metal >/dev/null 2>&1 \
    || fail "Metal compiler not found (xcrun --sdk macosx --find metal). Install Xcode and run: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
xcrun --sdk macosx --find metallib >/dev/null 2>&1 \
    || fail "metallib tool not found (xcrun --sdk macosx --find metallib)"

if command -v rustup >/dev/null 2>&1; then
    rustup target list --installed 2>/dev/null | grep -qx "$TARGET" \
        || fail "Rust target $TARGET is not installed; run: rustup target add $TARGET"
fi

printf 'Aurora Studio macOS build\n'
printf '  repo              : %s\n' "$ROOT"
printf '  host architecture : %s (build output is always %s)\n' "$HOST_ARCH" "$TARGET"
printf '  deployment target : %s\n' "$DEPLOYMENT_TARGET"
printf '  version           : %s\n' "$VERSION"
printf '  bundle id         : %s (local development only)\n' "$BUNDLE_ID"

# --- Build ------------------------------------------------------------------
printf '\n== cargo build ==\n'
cargo build -p "$PACKAGE_NAME" --release --locked --target "$TARGET"

TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
case "$TARGET_DIR" in
    /*) ;;
    *) TARGET_DIR="$ROOT/$TARGET_DIR" ;;
esac
BUILT_BIN="$TARGET_DIR/$TARGET/release/$BIN_NAME"
[ -f "$BUILT_BIN" ] || fail "cargo reported success but $BUILT_BIN does not exist"
[ -x "$BUILT_BIN" ] || fail "$BUILT_BIN is not executable"

# --- Assemble the bundle ----------------------------------------------------
printf '\n== assemble %s ==\n' "$APP_DIR"
rm -rf "$APP_DIR"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"

cp "$BUILT_BIN" "$CONTENTS/MacOS/$BIN_NAME"
chmod 0755 "$CONTENTS/MacOS/$BIN_NAME"

# Runtime assets (GPUI Kit assets, engine tables, portrait data) are compiled
# into the binary with rust-embed/include_bytes. The bundle therefore needs no
# asset tree; only the licence files are copied for inspection.
[ -f "$ROOT/LICENSE" ] || fail "missing $ROOT/LICENSE"
[ -f "$ROOT/ASSET-LICENSES.md" ] || fail "missing $ROOT/ASSET-LICENSES.md"
cp "$ROOT/LICENSE" "$CONTENTS/Resources/LICENSE"
cp "$ROOT/ASSET-LICENSES.md" "$CONTENTS/Resources/ASSET-LICENSES.md"

cat > "$CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!-- Local development bundle. Not a signing identity and not notarized. -->
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>$APP_NAME</string>
    <key>CFBundleExecutable</key>
    <string>$BIN_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <!--
      Finder / Open With association. Roles are Viewer and the handler rank is
      Alternate on purpose: the app must appear in "Open With" without claiming
      to be the default handler for any of these types. The extension lists are
      the union of crates/engine/src/raw.rs::is_raw and
      crates/engine/src/file_engine.rs::is_supported (plus HEIF, which the
      import dialog also accepts). No CFBundleURLTypes: the app registers no
      custom URL scheme, it only receives file:// URLs via
      application:openURLs: (crates/app/src/lib.rs on_open_urls).
    -->
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeName</key>
            <string>Camera RAW image</string>
            <key>CFBundleTypeRole</key>
            <string>Viewer</string>
            <key>LSHandlerRank</key>
            <string>Alternate</string>
            <key>CFBundleTypeExtensions</key>
            <array>
                <string>iiq</string>
                <string>nef</string>
                <string>cr3</string>
                <string>cr2</string>
                <string>arw</string>
                <string>dng</string>
                <string>raf</string>
                <string>orf</string>
                <string>rw2</string>
                <string>pef</string>
                <string>srw</string>
                <string>3fr</string>
                <string>fff</string>
                <string>mos</string>
            </array>
        </dict>
        <dict>
            <key>CFBundleTypeName</key>
            <string>Image</string>
            <key>CFBundleTypeRole</key>
            <string>Viewer</string>
            <key>LSHandlerRank</key>
            <string>Alternate</string>
            <key>CFBundleTypeExtensions</key>
            <array>
                <string>jpg</string>
                <string>jpeg</string>
                <string>png</string>
                <string>tif</string>
                <string>tiff</string>
                <string>webp</string>
                <string>heic</string>
                <string>heif</string>
            </array>
        </dict>
    </array>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>MacOSX</string>
    </array>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.photography</string>
    <key>LSMinimumSystemVersion</key>
    <string>$DEPLOYMENT_TARGET</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
</dict>
</plist>
PLIST

printf 'APPL????' > "$CONTENTS/PkgInfo"

# --- Verify the bundle ------------------------------------------------------
printf '\n== verify bundle ==\n'
plutil -lint "$CONTENTS/Info.plist"
[ -d "$CONTENTS/Resources" ] || fail "missing Contents/Resources"
[ -f "$CONTENTS/MacOS/$BIN_NAME" ] || fail "missing Contents/MacOS/$BIN_NAME"

ARCHS="$(lipo -archs "$CONTENTS/MacOS/$BIN_NAME")"
printf '  lipo -archs : %s\n' "$ARCHS"
case " $ARCHS " in
    *" arm64 "*) ;;
    *) fail "the built executable has no arm64 slice (archs: $ARCHS)" ;;
esac

file "$CONTENTS/MacOS/$BIN_NAME"

cat <<EOF

Bundle ready (unsigned):
  $APP_DIR

Next:
  ./scripts/package-macos.sh    # ad-hoc sign, verify, zip
Launch for manual testing:
  open "$APP_DIR"
EOF
