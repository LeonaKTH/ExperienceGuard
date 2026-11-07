# 架构与 Memory

`cli.py → app_cli.py → workflows/service.py` 是统一应用入口；`api/server.py` 调用同一服务对象，静态页面由包内 `api/` 资源提供。

## 取证与复用

1. `repository_tools/reader.py` 限制读取根目录、路径、类型和大小，记录 Git 身份与文件哈希，不导入用户代码。
2. `ci_analysis/analyzer.py` 解析失败日志、配置和 Python AST，生成观察事实、unknown、定位与确定性建议。
3. `storage/experience.py` 持久化领域记录，SQLite 事务与 outbox 配合 JSONL 追加；数据库触发器拒绝覆盖/删除经验，supersedes 表示替代。
4. `retrieval/retriever.py` 使用 BM25 召回固定候选；`features/development.py` 比较实际观察与历史谓词，版本不合法或信息缺失时保留 unknown。
5. `policies/selector.py` 执行策略。B4 要求历史成功、schema 一致和所有前提满足；B0 不使用经验。其他策略是显式对照或可选扩展，没有权重/数据时 B1/B5/New 不会假装可用。

记录 schema 仍使用原有领域版本标识，发行版本独立于记录格式。事实是配置/日志观察值，不自动代表当前已安装环境。

## Agent

`agents/development.py` 生成不可变披露计划，绑定源码、候选与输入摘要。默认每个文件只预览有限片段；需要更多内容时重新选择/审核，Agent 不能扩大读取范围。

`run_remote → agents/deepseek.configured_provider → DeepSeekJSONProvider.generate` 使用从 V2 整理的官方非 thinking JSON 协议。动作由宿主校验；模型须先 retrieve、check 后 select，应用层再次执行 B4 硬门槛。所有最终引用必须落在已返回的行区间内。

`Ledger.reserve → 请求 → Ledger.settle` 在持久文件锁下预留/结算。配置摘要绑定账本；不允许通过改配置静默重置。失败请求保留预留值，缺失用量时终止。Agent 循环另外限制步骤、请求、工具、Token 和墙钟时间。

`agents/protocol_test.py` 只用回环 HTTP 检查协议，明确标记 `mock_http_test`。`training/` 与 `evaluation/` 保留通用数据边界工具；没有分发训练好的选择器或任何收益结论。

## 验证与写回

自建示例只在与随包字节校验一致的副本中执行固定 unittest 命令；这不是运行任意第三方程序的隔离环境。真实仓库只读诊断，实际修改/验证由用户完成，记录为人工声明。Agent 输出本身不是验证，不能直接写成成功经验。
