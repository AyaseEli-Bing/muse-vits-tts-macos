import math
import os

# macOS 系统代理排除列表含 [::1]，会被 httpx 误解析导致 gradio 导入崩溃；本地服务直接绕过代理
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import torch
import gradio as gr
from torch import no_grad, LongTensor

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence

def _pick_device():
    # 默认优先 MPS（Apple GPU），检测不到则自动回退 CPU；VITS_DEVICE 可强制指定
    dev = (os.environ.get("VITS_DEVICE") or "mps").lower()
    if dev == "mps" and not torch.backends.mps.is_available():
        print("[设备] 未检测到 MPS，自动回退 CPU", flush=True)
        return "cpu"
    return dev

DEVICE = _pick_device()
DEVICE_LABEL = "Apple GPU (MPS)" if DEVICE == "mps" else "本机 CPU"
print(f"[设备] 推理设备 = {DEVICE}", flush=True)
# Apple 芯片 CPU 推理：8 线程已到收益拐点（10 线程不再更快）；MPS 下此项影响小但无害
torch.set_num_threads(int(os.environ.get("VITS_THREADS", min(8, os.cpu_count() or 4))))
BASE = os.path.dirname(os.path.abspath(__file__))
# 优先仓库内 models_local/（发布后可自包含）；兼容把权重放在仓库同级 ../models_local 的开发布局
_MODEL_DIR_CANDIDATES = [os.path.join(BASE, "models_local"), os.path.join(BASE, "..", "models_local")]
MODEL_DIR = next((p for p in _MODEL_DIR_CANDIDATES if os.path.isdir(p)), _MODEL_DIR_CANDIDATES[0])

MODELS = {
    "tricolor（中日 12 人）": {
        "config": "tricolor_config.json",
        "model": "tricolor.pth",
        "languages": {"简体中文": "[ZH]", "日本語": "[JA]", "Mix": ""},
    },
    "muse9（µ's 9 人 · 日文）": {
        "config": "muse9_config.json",
        "model": "muse9.pth",
        "languages": {"Japanese": ""},
    },
}

_cache = {}


def load_model(key):
    if key in _cache:
        return _cache[key]
    info = MODELS[key]
    hps = utils.get_hparams_from_file(os.path.join(MODEL_DIR, info["config"]))
    net = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(DEVICE)
    net.eval()
    utils.load_checkpoint(os.path.join(MODEL_DIR, info["model"]), net, None)
    # 预热一次极小前向：触发 MPS kernel 编译，消除首次合成 ~1s 的高延迟
    with no_grad():
        dummy = LongTensor([[1] * 8]).to(DEVICE)
        net.infer(dummy, LongTensor([8]).to(DEVICE), sid=LongTensor([0]).to(DEVICE))
    speakers = hps.speakers
    if isinstance(speakers, dict):
        speakers = list(speakers.keys())
    _cache[key] = (hps, net, speakers)
    return _cache[key]


def get_text(text, hps):
    norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        norm = commons.intersperse(norm, 0)
    return LongTensor(norm)


def refresh(model_key):
    _, _, speakers = load_model(model_key)
    langs = list(MODELS[model_key]["languages"].keys())
    return (
        gr.update(choices=langs, value=langs[0]),
        gr.update(choices=speakers, value=speakers[0]),
    )


def synthesize(model_key, language, speaker, text, noise_scale, noise_scale_w, length_scale):
    if not text or not text.strip():
        raise gr.Error("请输入要合成的文本")
    hps, net, speakers = load_model(model_key)
    mark = MODELS[model_key]["languages"][language]
    text = mark + text.strip() + mark
    sid = speakers.index(speaker)
    stn = get_text(text, hps)
    with no_grad():
        x = stn.unsqueeze(0).to(DEVICE)
        xl = LongTensor([stn.size(0)]).to(DEVICE)
        s = LongTensor([sid]).to(DEVICE)
        audio = net.infer(
            x, xl, sid=s,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=1.0 / length_scale,
        )[0][0, 0].data.cpu().float().numpy()
    return (hps.data.sampling_rate, audio)


def claude_spark(size=46, color="#D97757"):
    lengths = [47, 29, 43, 27, 47, 29, 43, 27, 47, 29, 43, 27]
    rays = []
    for i, r in enumerate(lengths):
        ang = math.radians(i * 30 - 90)
        x1, y1 = 50 + 11 * math.cos(ang), 50 + 11 * math.sin(ang)
        x2, y2 = 50 + r * math.cos(ang), 50 + r * math.sin(ang)
        rays.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="7.5" stroke-linecap="round"/>'
        )
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">{"".join(rays)}</svg>'
    )


claude_theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#FBF1EC", c100="#F6DFD4", c200="#EDC1AC", c300="#E4A285",
        c400="#DC8C6C", c500="#D97757", c600="#C6613F", c700="#A84E32",
        c800="#8A3F29", c900="#6E3222", c950="#4A2116",
    ),
    secondary_hue=gr.themes.Color(
        c50="#FAF9F5", c100="#F5F4EF", c200="#E5E4DF", c300="#DAD9D3",
        c400="#B9B8B1", c500="#87867F", c600="#6B6A63", c700="#4D4C46",
        c800="#3D3929", c900="#2A2823", c950="#1F1E1D",
    ),
    neutral_hue=gr.themes.Color(
        c50="#FAF9F5", c100="#F5F4EF", c200="#E5E4DF", c300="#DAD9D3",
        c400="#B9B8B1", c500="#87867F", c600="#6B6A63", c700="#4D4C46",
        c800="#3D3929", c900="#2A2823", c950="#1F1E1D",
    ),
    font=["-apple-system", "BlinkMacSystemFont", "Söhne", "Segoe UI", "system-ui", "sans-serif"],
    font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
).set(
    body_background_fill="#F5F4EF",
    body_background_fill_dark="#1F1E1D",
    body_text_color="#3D3929",
    body_text_color_dark="#ECEAE3",
    body_text_color_subdued="#87867F",
    background_fill_primary="#F5F4EF",
    background_fill_secondary="#FAF9F5",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#2A2823",
    block_border_color="#E5E4DF",
    block_border_width="1px",
    block_label_background_fill="#FFFFFF",
    block_label_text_color="#6B6A63",
    block_title_text_color="#3D3929",
    block_shadow="0 1px 2px rgba(61, 57, 41, 0.05)",
    block_radius="14px",
    border_color_primary="#E5E4DF",
    border_color_accent="#D97757",
    color_accent="#D97757",
    color_accent_soft="#FBF1EC",
    button_primary_background_fill="#D97757",
    button_primary_background_fill_hover="#C6613F",
    button_primary_border_color="#D97757",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F5F4EF",
    button_secondary_border_color="#DAD9D3",
    button_secondary_text_color="#3D3929",
    button_large_radius="12px",
    button_small_radius="10px",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#2A2823",
    input_border_color="#DAD9D3",
    input_border_color_focus="#D97757",
    input_placeholder_color="#B9B8B1",
    input_radius="12px",
    slider_color="#D97757",
    table_odd_background_fill="#FAF9F5",
    table_even_background_fill="#FFFFFF",
)

CLAUDE_CSS = """
body { background: #F5F4EF !important; --bg: #F5F4EF; }
.gradio-container { max-width: 860px !important; margin: 0 auto !important; }
.claude-header { text-align: center; padding: 30px 0 14px; }
.claude-spark { display: flex; justify-content: center; margin-bottom: 14px; }
.claude-title {
    font-family: "Tiempos Headline", "Copernicus", "Songti SC", "Noto Serif SC", Georgia, serif;
    font-weight: 500; font-size: 2.1rem; line-height: 1.2;
    color: #3D3929; letter-spacing: -0.01em; margin: 0;
}
.claude-sub { color: #87867F; font-size: 0.95rem; margin: 10px 0 0; }
.dark .claude-title { color: #ECEAE3; }
.dark .claude-sub { color: #B9B8B1; }
footer { display: none !important; }
.built-with { display: none !important; }
.prose { font-size: 0.95rem; }
"""

# Claude 标志性外观是暖米色浅色界面；覆盖系统深色偏好，始终锁定浅色主题
FORCE_LIGHT_HEAD = r"""
<script>
(function () {
  var real = window.matchMedia.bind(window);
  window.matchMedia = function (query) {
    var mql = real(query);
    if (query.replace(/\s/g, '') === '(prefers-color-scheme:dark)') {
      try { Object.defineProperty(mql, 'matches', { get: function () { return false; } }); } catch (e) {}
    }
    return mql;
  };
  function stripDark() {
    if (document.body && document.body.classList.contains('dark')) {
      document.body.classList.remove('dark');
    }
  }
  function start() {
    stripDark();
    new MutationObserver(stripDark).observe(document.body, { attributes: true, attributeFilter: ['class'] });
  }
  if (document.body) { start(); } else { document.addEventListener('DOMContentLoaded', start); }
})();
</script>
"""

header_html = (
    '<div class="claude-header">'
    f'<div class="claude-spark">{claude_spark(46)}</div>'
    '<h1 class="claude-title">语音合成</h1>'
    f'<p class="claude-sub">本地 VITS · 离线运行于{DEVICE_LABEL} · 选择模型与说话人，输入文本即可合成</p>'
    '</div>'
)

with gr.Blocks(title="VITS 本地语音合成", theme=claude_theme, css=CLAUDE_CSS, head=FORCE_LIGHT_HEAD) as demo:
    gr.HTML(header_html)
    with gr.Row():
        model_dd = gr.Dropdown(list(MODELS.keys()), value=list(MODELS.keys())[0], label="模型")
        lang_dd = gr.Dropdown(choices=[], label="语言")
        spk_dd = gr.Dropdown(choices=[], label="说话人")
    text_box = gr.Textbox(label="文本", lines=4, placeholder="输入要合成的文本…")
    with gr.Accordion("高级参数", open=False):
        with gr.Row():
            ns = gr.Slider(0.0, 1.0, value=0.4, step=0.01, label="情感变化 (noise_scale)")
            nsw = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="音素时长 (noise_scale_w)")
            ls = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="语速 (越大越快)")
    btn = gr.Button("合成语音", variant="primary", size="lg")
    out_audio = gr.Audio(label="合成结果", type="numpy")

    model_dd.change(refresh, inputs=model_dd, outputs=[lang_dd, spk_dd])
    btn.click(
        synthesize,
        inputs=[model_dd, lang_dd, spk_dd, text_box, ns, nsw, ls],
        outputs=out_audio,
    )
    demo.load(refresh, inputs=model_dd, outputs=[lang_dd, spk_dd])

if __name__ == "__main__":
    # 启动即预加载全部模型，消除首次合成的 ~1s 权重加载等待
    for _key in MODELS:
        try:
            load_model(_key)
            print(f"[预加载] {_key} 就绪", flush=True)
        except Exception as _e:
            print(f"[预加载失败] {_key}: {_e!r}", flush=True)
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False)