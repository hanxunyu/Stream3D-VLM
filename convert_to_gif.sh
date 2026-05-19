#!/usr/bin/env bash
# 将指定的 mp4 视频转换为高质量 GIF。
# 使用 ffmpeg 两步法（palettegen + paletteuse），相比单步转换可显著提升色彩质量、减少抖动。
#
# 可通过环境变量调整参数：
#   FPS    - 输出 GIF 帧率（默认 15）
#   WIDTH  - 输出 GIF 宽度（高度按比例自动计算；默认 960）
#   DITHER - 抖动算法（默认 sierra2_4a，可选：none / bayer / sierra2 / sierra2_4a / floyd_steinberg）
#   STATS  - palettegen 统计模式（默认 diff，更适合运动画面；也可用 full）
#
# 用法：
#   bash convert_to_gif.sh
#   FPS=20 WIDTH=1280 bash convert_to_gif.sh

set -euo pipefail

FPS="${FPS:-15}"
WIDTH="${WIDTH:-960}"
DITHER="${DITHER:-sierra2_4a}"
STATS="${STATS:-diff}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG_DIR="${SCRIPT_DIR}/images"

INPUTS=(
    "${IMG_DIR}/realtime_scene0200_00_converted.mp4"
    "${IMG_DIR}/forward_scene0599_02_converted.mp4"
    "${IMG_DIR}/backward_scene0426_00_converted.mp4"
)

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "错误：未找到 ffmpeg，请先安装（macOS: brew install ffmpeg）" >&2
    exit 1
fi

convert_one() {
    local input="$1"
    local base
    base="$(basename "${input%.*}")"
    local palette="${IMG_DIR}/${base}.palette.png"
    local output="${IMG_DIR}/${base}.gif"

    if [[ ! -f "$input" ]]; then
        echo "跳过：未找到 $input" >&2
        return 0
    fi

    echo "==> 处理：$input"
    echo "    参数: fps=${FPS}, width=${WIDTH}, dither=${DITHER}, stats=${STATS}"

    # 第一步：生成专属调色板（256 色），针对运动画面使用 stats_mode=diff
    ffmpeg -hide_banner -loglevel error -y \
        -i "$input" \
        -vf "fps=${FPS},scale=${WIDTH}:-2:flags=lanczos,palettegen=stats_mode=${STATS}" \
        "$palette"

    # 第二步：使用调色板生成 GIF，并应用抖动算法
    ffmpeg -hide_banner -loglevel error -y \
        -i "$input" -i "$palette" \
        -lavfi "fps=${FPS},scale=${WIDTH}:-2:flags=lanczos[x];[x][1:v]paletteuse=dither=${DITHER}" \
        -loop 0 \
        "$output"

    rm -f "$palette"

    local size
    size="$(du -h "$output" | cut -f1)"
    echo "    完成 -> $output (${size})"
}

for input in "${INPUTS[@]}"; do
    convert_one "$input"
done

echo "全部完成。"
