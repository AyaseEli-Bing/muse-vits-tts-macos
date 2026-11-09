# muse-vits-tts-macos — 本地 VITS 语音合成（macOS / Apple Silicon）

在 Mac（M 系列芯片）上**离线**运行的 VITS 语音合成：**Gradio 网页界面**（`webui.py`）+ 命令行入口（`cmd_inference.py`）。默认用 **MPS（Apple GPU）** 推理，检测不到自动回退 CPU；界面为 Claude 风格。

> **许可与来源**：推理代码框架派生自 **[Plachtaa/VITS-fast-fine-tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning)**（**Apache-2.0**，见 [`LICENSE`](./LICENSE)，原始说明保留于 [`README_UPSTREAM.md`](./README_UPSTREAM.md)）。**本仓库不包含任何模型权重**；权重来源与再分发权由使用者自行负责。

---

## 1. 前置要求
- macOS on **Apple Silicon（arm64）**
- **Python 3.11**（⚠️ 勿用 3.12：`pyopenjtalk-prebuilt` 无 cp312 arm64 轮子、且缺 `distutils` 构建失败）
  - `brew install python@3.11` 或 `uv python install 3.11`
- Xcode Command Line Tools（编译 `monotonic_align`）：`xcode-select --install`

## 2. 一键安装
```bash
bash setup.sh
```
建 `.venv`(3.11) → 装 `setuptools<81` 与 `requirements-mac.txt` → 建嵌套包目录并就地编译 `monotonic_align`。
可用 `PY=... PIP_MIRROR=... bash setup.sh` 覆盖（大陆网络默认走清华 PyPI 镜像）。

## 3. 放置模型权重（不含在本仓库）
把权重与 config 放进 [`models_local/`](./models_local)（详见其内 README），并在 `webui.py` 的 `MODELS` 字典登记文件名、语言标记：
```python
MODELS = {
  "中日 12 人示例": {
    "config": "xxx_config.json", "model": "xxx.pth",
    "languages": {"简体中文": "[ZH]", "日本語": "[JA]", "Mix": ""},
  },
}
```
`languages` 的标记须与模型训练的 `text_cleaners` 匹配；说话人从 config 的 `speakers` 自动读取。

## 4. 运行
```bash
source .venv/bin/activate
python webui.py          # → http://127.0.0.1:7860
```
命令行：
```bash
python cmd_inference.py -m models_local/xxx.pth -c models_local/xxx_config.json \
  -t "你好，世界" -l 简体中文 -s "<说话人>" -o output
```

## 5. 设备与线程（环境变量）
| 变量 | 默认 | 说明 |
|---|---|---|
| `VITS_DEVICE` | `mps`（无则 `cpu`） | 推理设备；可 `VITS_DEVICE=cpu` 强制 |
| `VITS_THREADS` | `min(8, 核数)` | CPU 线程数（MPS 下影响小） |

启动会**预加载并预热**模型，消除首击冷启动；实测短句往返 MPS ~275ms、CPU@8 ~495ms。

## 6. 已知注意事项
- **macOS 系统代理**：`NO_PROXY=*` 已在 `webui.py` 顶部（导入 gradio 前）设置，绕过 `[::1]` 触发 httpx 崩溃。勿删。
- **`setuptools<81`**：`pyopenjtalk` 依赖 `pkg_resources`。
- **默认参数**：`noise_scale=0.4` / `noise_scale_w=0.3` / 语速 `1.0`（干净稳定向）；每次点「合成」重新采样，可多生成挑最佳。

## 7. 目录速览
```
webui.py              # Gradio 界面（Claude 风格、MPS/CPU、预加载预热）
cmd_inference.py      # 命令行合成
models_local/         # 放你的权重+config（本仓库不含权重）
setup.sh / requirements-mac.txt   # macOS 可复现环境
monotonic_align/      # 源码；由 setup.sh 编译
LICENSE / README_UPSTREAM.md      # Apache-2.0 与上游说明
```
