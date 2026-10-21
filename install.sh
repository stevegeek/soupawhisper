#!/bin/bash
# Build SoupaWhisper and install it as a systemd user service.
#
#   ./install.sh             CUDA build (CUDA toolkit in /opt/cuda)
#   ./install.sh --vulkan    Vulkan build
#   ./install.sh --cpu       CPU-only build
#   ./install.sh --uninstall remove the binary and the service (keeps config and models)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HOME/.local/bin/soupawhisper"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/soupawhisper.service"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/soupawhisper"

# The service briefly shipped as soupawhisper-rs while the Python version still existed
remove_legacy_unit() {
    if [ -f "$UNIT_DIR/soupawhisper-rs.service" ]; then
        systemctl --user disable --now soupawhisper-rs 2>/dev/null || true
        rm -f "$UNIT_DIR/soupawhisper-rs.service" "$HOME/.local/bin/soupawhisper-rs"
    fi
}

if [ "${1:-}" = "--uninstall" ]; then
    systemctl --user disable --now soupawhisper 2>/dev/null || true
    remove_legacy_unit
    rm -f "$UNIT" "$BIN"
    systemctl --user daemon-reload
    echo "Removed $BIN and $UNIT (config in $CONFIG_DIR and models in ~/.local/share/soupawhisper kept)"
    exit 0
fi

FEATURES="cuda"
case "${1:-}" in
    --cpu) FEATURES="" ;;
    --vulkan) FEATURES="vulkan" ;;
    "") ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
esac

missing=()
for cmd in cargo cmake arecord; do
    command -v "$cmd" >/dev/null || missing+=("$cmd")
done
if [ "${XDG_SESSION_TYPE:-}" = "wayland" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    for cmd in wl-copy ydotool; do command -v "$cmd" >/dev/null || missing+=("$cmd"); done
else
    for cmd in xclip xdotool; do command -v "$cmd" >/dev/null || missing+=("$cmd"); done
fi
if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing: ${missing[*]} (see README.md, Requirements)" >&2
    exit 1
fi

cd "$DIR"
[ "$FEATURES" = "cuda" ] && export PATH="/opt/cuda/bin:$PATH"
cargo build --release --no-default-features ${FEATURES:+--features "$FEATURES"}

mkdir -p "$(dirname "$BIN")" "$UNIT_DIR" "$CONFIG_DIR"
remove_legacy_unit
install -m 755 target/release/soupawhisper "$BIN"
if [ ! -f "$CONFIG_DIR/config.ini" ]; then
    cp config.example.ini "$CONFIG_DIR/config.ini"
    echo "Created $CONFIG_DIR/config.ini"
fi

cat > "$UNIT" <<EOF
[Unit]
Description=SoupaWhisper Voice Dictation
After=graphical-session.target
Wants=ydotool.service

[Service]
Type=simple
ExecStart=$BIN
Restart=on-failure
RestartSec=5
Environment=XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-wayland}
Environment=WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
Environment=DISPLAY=${DISPLAY:-:0}

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
echo "Installed $BIN and $UNIT"
echo "Start it now and on login:  systemctl --user enable --now soupawhisper"
