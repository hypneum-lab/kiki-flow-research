#!/usr/bin/env bash
# Build espeak-ng from the pinned upstream source archive into a prefix.
#
# CI/test-only: phonemizer (used by HeuristicLabeler) needs the
# libespeak-ng shared library. No brew/apt: the exact upstream release
# is downloaded, its SHA-256 verified (abort on mismatch), and built with
# CMake without optional network-fetched components (sonic, pcaudio,
# mbrola). espeak-ng is GPL-3.0-or-later; it is used in CI only and is
# not redistributed.
#
# Usage: build_espeak_ng.sh <prefix>
set -euo pipefail

ESPEAK_NG_VERSION="1.52.0"
ESPEAK_NG_URL="https://github.com/espeak-ng/espeak-ng/archive/refs/tags/${ESPEAK_NG_VERSION}.tar.gz"
ESPEAK_NG_SHA256="bb4338102ff3b49a81423da8a1a158b420124b055b60fa76cfb4b18677130a23"

prefix="${1:?usage: build_espeak_ng.sh <prefix>}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

curl --fail --silent --show-error --location \
    --output "$work/espeak-ng.tar.gz" "$ESPEAK_NG_URL"
echo "${ESPEAK_NG_SHA256}  $work/espeak-ng.tar.gz" | shasum -a 256 -c -

tar -xzf "$work/espeak-ng.tar.gz" -C "$work"
src="$work/espeak-ng-${ESPEAK_NG_VERSION}"

# SONIC_LIB/SONIC_INC are set to dummies so deps.cmake does not
# FetchContent sonic from git; USE_LIBSONIC=OFF keeps it unlinked.
cmake -S "$src" -B "$work/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DBUILD_SHARED_LIBS=ON \
    -DSONIC_LIB=unused -DSONIC_INC=unused -DUSE_LIBSONIC=OFF \
    -DUSE_LIBPCAUDIO=OFF -DUSE_MBROLA=OFF -DUSE_ASYNC=OFF \
    -DESPEAK_BUILD_MANPAGES=OFF
cmake --build "$work/build" --parallel
cmake --install "$work/build"

"$prefix/bin/espeak-ng" --version
