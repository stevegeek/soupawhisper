#!/bin/bash
# Build the Rust port and install it as the soupawhisper-rs user service.
# It reads the same ~/.config/soupawhisper/config.ini as the Python version and
# conflicts with soupawhisper.service, so starting one stops the other.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HOME/.local/bin/soupawhisper-rs"
UNIT="$HOME/.config/systemd/user/soupawhisper-rs.service"

FEATURES="cuda"
[ "${1:-}" = "--cpu" ] && FEATURES=""
[ "${1:-}" = "--vulkan" ] && FEATURES="vulkan"

cd "$DIR"
if [ "$FEATURES" = "cuda" ]; then
    export PATH="/opt/cuda/bin:$PATH"
fi
cargo build --release --no-default-features ${FEATURES:+--features "$FEATURES"}

mkdir -p "$(dirname "$BIN")" "$(dirname "$UNIT")"
install -m 755 target/release/soupawhisper "$BIN"

cat > "$UNIT" <<EOF
[Unit]
Description=SoupaWhisper Voice Dictation (Rust)
After=graphical-session.target
Wants=ydotool.service
Conflicts=soupawhisper.service

[Service]
Type=simple
ExecStart=$BIN
Restart=on-failure
RestartSec=5
Environment=XDG_SESSION_TYPE=wayland
Environment=WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
echo "Installed $BIN and $UNIT"
echo "Switch over:  systemctl --user disable --now soupawhisper && systemctl --user enable --now soupawhisper-rs"
echo "Switch back:  systemctl --user disable --now soupawhisper-rs && systemctl --user enable --now soupawhisper"
