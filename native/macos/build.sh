#!/bin/sh
set -eu
source_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build_dir=${1:-"$source_dir/build"}
app="$build_dir/VisionLab.app"
mkdir -p "$app/Contents/MacOS"
xcrun clang -fobjc-arc -fmodules -c "$source_dir/ViewController.m" -o "$build_dir/ViewController.o"
xcrun clang++ "$source_dir/Main.cpp" "$build_dir/ViewController.o" -framework Cocoa -framework AVFoundation -framework Vision -framework QuartzCore -o "$app/Contents/MacOS/VisionLab"
cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>VisionLab</string>
<key>CFBundleIdentifier</key><string>com.zachmartin.visionlab</string>
<key>CFBundleName</key><string>Vision Lab</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>NSCameraUsageDescription</key><string>Track rectangular surfaces locally with the camera.</string>
<key>NSPrincipalClass</key><string>NSApplication</string>
</dict></plist>
PLIST
printf 'Built %s\n' "$app"
