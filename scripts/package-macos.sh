#!/usr/bin/env bash
# Verify an Aurora Studio .app bundle, apply an ad-hoc signature, and zip it.
#
# This does NOT notarize the app and does NOT claim a Developer ID identity.
# Gatekeeper will still block the zip on other machines unless the user
# explicitly allows it; that is expected for a local development build.
set -euo pipefail
umask 022

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

warn() {
    printf 'warning: %s\n' "$*" >&2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

APP_NAME="Aurora Studio"
BIN_NAME="aurora-studio"
APP_DIR="$ROOT/dist/$APP_NAME.app"
INFO_PLIST="$APP_DIR/Contents/Info.plist"
EXECUTABLE="$APP_DIR/Contents/MacOS/$BIN_NAME"

if [ "$(uname -s)" != "Darwin" ]; then
    fail "this script verifies a macOS .app and must run on macOS"
fi
[ -d "$APP_DIR" ] || fail "missing $APP_DIR; run ./scripts/build-macos.sh first"
[ -f "$INFO_PLIST" ] || fail "missing $INFO_PLIST"
[ -f "$EXECUTABLE" ] || fail "missing $EXECUTABLE"
[ -x "$EXECUTABLE" ] || fail "$EXECUTABLE is not executable"

for tool in plutil file lipo otool codesign ditto shasum; do
    command -v "$tool" >/dev/null 2>&1 || fail "$tool is not on PATH"
done

VERSION="$(plutil -extract CFBundleShortVersionString raw -o - "$INFO_PLIST" 2>/dev/null || true)"
[ -n "$VERSION" ] || fail "Info.plist has no CFBundleShortVersionString"

printf '== Info.plist ==\n'
plutil -lint "$INFO_PLIST"
printf '  version : %s\n' "$VERSION"
printf '  bundle  : %s\n' "$(plutil -extract CFBundleIdentifier raw -o - "$INFO_PLIST")"
printf '  minimum : %s\n' "$(plutil -extract LSMinimumSystemVersion raw -o - "$INFO_PLIST")"

printf '\n== executable format ==\n'
file "$EXECUTABLE"

ARCHS="$(lipo -archs "$EXECUTABLE")"
printf '  lipo -archs : %s\n' "$ARCHS"
case " $ARCHS " in
    *" arm64 "*) ;;
    *) fail "the executable has no arm64 slice (archs: $ARCHS)" ;;
esac
case " $ARCHS " in
    *" x86_64 "*) warn "the executable also contains x86_64; record Intel testing separately, do not infer it from this ARM64 build" ;;
esac

# --- Linkage ----------------------------------------------------------------
# System libraries and frameworks are expected. Anything else must either be
# bundled inside Contents/Frameworks (or Contents/MacOS) or be reported as a
# packaging defect: the app must not depend on Homebrew or a developer path.
printf '\n== otool -L ==\n'
LINKAGE="$(otool -L "$EXECUTABLE" | tail -n +2)"
printf '%s\n' "$LINKAGE"

non_system=0
while IFS= read -r line; do
    [ -n "$line" ] || continue
    lib="${line%% (compatibility*}"
    lib="${lib#"${lib%%[![:space:]]*}"}"
    [ -n "$lib" ] || continue
    case "$lib" in
        /System/Library/*|/usr/lib/*)
            ;;
        @rpath/*|@loader_path/*|@executable_path/*)
            name="${lib##*/}"
            if [ -f "$APP_DIR/Contents/Frameworks/$name" ] || [ -f "$APP_DIR/Contents/MacOS/$name" ]; then
                printf '  bundled non-system dependency: %s\n' "$lib"
            else
                printf '  NON-SYSTEM DEPENDENCY NOT BUNDLED: %s\n' "$lib" >&2
                non_system=1
            fi
            ;;
        *)
            printf '  NON-SYSTEM DEPENDENCY: %s\n' "$lib" >&2
            non_system=1
            ;;
    esac
done <<< "$LINKAGE"

if [ "$non_system" -ne 0 ]; then
    fail "the executable links against a non-system dylib that is not inside the bundle; vendor it under Contents/Frameworks with an @rpath install name, or fix the dependency"
fi

# --- Ad-hoc signature -------------------------------------------------------
printf '\n== ad-hoc codesign ==\n'
codesign --force --sign - --timestamp=none "$APP_DIR"
codesign --verify --strict --verbose=2 "$APP_DIR"

SIGNATURE="$(codesign -dv --verbose=4 "$APP_DIR" 2>&1 | sed -n 's/^Signature=//p' | head -n 1)"
printf '  Signature : %s\n' "${SIGNATURE:-unknown}"
[ "$SIGNATURE" = "adhoc" ] || fail "expected an ad-hoc signature, got '${SIGNATURE:-none}'"
printf '  This is an ad-hoc signature. The app is NOT notarized.\n'

# --- Archive ----------------------------------------------------------------
printf '\n== zip ==\n'
ZIP="$ROOT/dist/Aurora Studio-$VERSION-arm64.zip"
rm -f "$ZIP" "$ZIP.sha256"
ditto -c -k --keepParent --sequesterRsrc "$APP_DIR" "$ZIP"
[ -s "$ZIP" ] || fail "ditto did not produce $ZIP"

( cd "$ROOT/dist" && shasum -a 256 "$(basename "$ZIP")" ) | tee "$ZIP.sha256"
printf '  archive : %s\n' "$ZIP"
printf '  checksum: %s\n' "$ZIP.sha256"

cat <<EOF

Verification passed.
  app  : $APP_DIR
  zip  : $ZIP
  sign : ad-hoc only; not notarized; no Developer ID identity used.

Not verified by this script:
  * launching the app from Finder
  * launching on a clean Mac without Xcode/Homebrew
  * Gatekeeper acceptance on another machine
EOF
