# 使用说明

先完成 README 安装、`init` 和离线工作台演示。Git 可用时自建示例会创建本地提交；没有 Git 时部分提交证据不可用。

## 远程 CLI

使用自己管理的进程环境配置好 Key、显式开关与本地价格/限额文件后：

```powershell
$report = & $egPython -m experienceguard diagnose --data-dir $egData --fixture source | ConvertFrom-Json
$preview = & $egPython -m experienceguard agent-preview --data-dir $egData --case-id $report.case_id --file worker.py --file compatlib.py --file pyproject.toml | ConvertFrom-Json
$preview | ConvertTo-Json -Depth 30

# 只在审核完整披露范围、确认服务配置和费用后执行
& $egPython -m experienceguard agent-run --data-dir $egData --case-id $report.case_id --plan-id $preview.plan_id --consent-sha $preview.plan_sha256 --allow-remote --acknowledge-upload --confirm-cost
```

不希望调用模型时可使用 `agent-protocol-test --case-id ... --plan-id ...`，它只启动本机脚本化 HTTP 响应端，不是 LLM。

远程请求/响应、诊断和验证记录均在 `--data-dir` 指定的本地目录，不应复制到公开仓库。关掉终端或安全入口启动的服务会结束其进程内凭证使用。

## 自己的仓库

启动 Web 时额外设置 `--allow-root ../sample-repo`。仅授权需要诊断的具体目录，避免整个用户目录。选中文件前先检查内容，远程预览不会自动涵盖完整源文件。配置改变或源码更新后重新诊断和预览。

人工验证结果必须提供实际日志；失败与结论不足也要保留。自建示例自动验证不能扩展为对任意仓库执行测试。

## 依赖

锁文件记录 Windows x64 / CPython 3.13 下的版本，包含开发测试工具。Python 包声明支持 3.11—3.13，其他平台应按 pyproject 解析兼容依赖；示例和测试的主要验收环境是 Python 3.13。默认不需要 ToolSandbox、Torch、CUDA 或 Embedding 权重。

发行包同时包含 Web 静态文件与自建示例，可通过普通 wheel 安装运行。没有再分发公开第三方故障语料，工作台相应入口会隐藏。`public-evaluate` 仅供使用者对自行合法准备的可选数据执行静态诊断核查。
