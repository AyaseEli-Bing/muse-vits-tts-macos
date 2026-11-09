#!/usr/bin/env bash
# macOS (Apple Silicon) 一键部署 VITS TTS 推理环境
# 用法： bash setup.sh   （可用 PY=python3.11 / PIP_MIRROR=<url> 覆盖）
set -euo pipefail

PY="${PY:-python3.11}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# 1) 虚拟环境（务必 3.11；3.12 缺 pyopenjtalk cp311 轮子且无 distutils）
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "找不到 $PY。请先安装，例如：brew install python@3.11  或  uv python install 3.11" >&2
  exit 1
fi
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
python -m pip install "setuptools<81" -i "$PIP_MIRROR"

# 2) 依赖
python -m pip install -r requirements-mac.txt -i "$PIP_MIRROR"

# 3) 编译 monotonic_align（Cython 扩展）。__init__.py 从嵌套包路径导入，
#    需先建嵌套目录再就地构建，否则报 “could not create .../core...so”。
mkdir -p monotonic_align/monotonic_align
( cd monotonic_align && python setup.py build_ext --inplace )

echo ""
echo "环境就绪。下一步："
echo "  1) 把你的 .pth 权重与对应 config.json 放到仓库内 models_local/ （见 README_MACOS.md）"
echo "  2) 编辑 webui.py 的 MODELS 字典，填入你的 config/model 文件名、语言标记与说话人"
echo "  3) 运行： source .venv/bin/activate && python webui.py  → http://127.0.0.1:7860"
