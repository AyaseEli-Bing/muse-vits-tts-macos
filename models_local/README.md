# models_local/ —— 模型权重放置目录

本仓库**不含模型权重**（体积大，且再分发权需自行确认）。把你要用的 VITS checkpoint 与对应 config 放到这里：

```
models_local/
├── <你的>.pth                # 模型权重
└── <你的>_config.json        # 该模型的 config
```

然后在仓库根 `webui.py` 的 `MODELS` 字典里登记：

```python
MODELS = {
  "显示名": {
    "config": "<你的>_config.json",   # 本目录下的文件名
    "model":  "<你的>.pth",
    "languages": {"简体中文": "[ZH]", "日本語": "[JA]", "Mix": ""},
  },
}
```

- `languages` 的 value 是加在文本前后的语言标记，须与模型训练时的 `text_cleaners` 匹配：
  `japanese_cleaners` 用 `""`；`zh_ja_mixture_cleaners` 用 `[ZH]` / `[JA]` / `[EN]` / `""`(Mix)。
- 说话人列表从 config 的 `speakers` 自动读取（list 或 dict 皆可）。

> 除 `README.md` 外，本目录内容已被 `.gitignore` 忽略，不会被提交。
