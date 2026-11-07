# Third-party notices

ExperienceGuard 原创应用、适配代码和自建示例采用根目录 MIT 许可。BM25、AST、版本比较、逻辑回归等为已有技术，不主张原创算法。

| 组件 | 用途 | 许可与分发方式 |
|---|---|---|
| NumPy / SciPy | 数值计算和可选训练 | BSD-3-Clause；由 pip 安装，未将其源码或二进制打包进仓库 |
| scikit-learn | 可选轻量分类与校准 | BSD-3-Clause；同上 |
| PyYAML | 配置 | MIT；同上 |
| jsonschema | JSON 动作校验 | MIT；同上 |
| filelock | 本地预算与追加操作锁 | Unlicense；同上 |
| packaging | 版本与依赖谓词 | Apache-2.0 / BSD-2-Clause 双许可；同上 |
| pytest / pytest-cov / coverage / Ruff / setuptools 及传递依赖 | 安装与开发工具 | 以各安装包 LICENSE/NOTICE 为准；锁文件不重新许可这些依赖 |
| ONNX Runtime / Hugging Face tokenizers | 可选本地 Embedding 接口 | MIT / Apache-2.0；默认不安装、不下载权重 |

`benchmarks/toolsandbox/adapter.py` 是保留在通用序列化代码依赖闭包内的本项目适配层，非默认应用后端。原始 Apple ToolSandbox 固定提交为 `c8571d7854316d2e1c5f288e59fe1e34e53f6dd1`。仓库不分发或安装其实现，原始 [LICENSE](https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/LICENSE) 与 [ACKNOWLEDGEMENTS](https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/ACKNOWLEDGEMENTS) 留存在 `third_party/`，不将其改为 MIT，也不表示其作者为本项目背书。

DeepSeek 是可选远程服务，使用者需遵守其服务条款；不分发任何模型权重。用户导入仓库、日志及本地数据库不是本项目的开源分发内容。第三方公开故障源码、帖子文本、真实模型请求/响应均未随此公开版本分发。
