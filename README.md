# 降压线连接知识库离线 ETL

第一阶段只做离线建库：从飞书多维表格和飞书群聊历史消息抽取行车记录仪降压线接线知识，写入 SQLite 结构化表和 SQLite embedding 表。在线客服、webhook、DeepSeek 回复生成、人工反馈闭环不在本阶段实现。

## 项目结构

- `etl.py`：主 CLI，支持 `table` 和 `chat` 两个子命令。
- `etl_kb/feishu.py`：飞书 API 客户端，包含 tenant token、Base 分页、群聊消息分页、图片 token 下载。
- `etl_kb/llm.py`：GLM 视觉、GLM 结构化抽取、GLM embedding，以及 OpenAI 兼容中转兜底。
- `etl_kb/segmenter.py`：群聊消息按时间窗口和话题相关性切分成案例片段。
- `etl_kb/vehicle_normalizer.py`：车型品牌别名归一化和年份代际匹配。
- `etl_kb/storage.py`：SQLite 建表、写入、车辆查询、brute-force 余弦向量检索。
- `configs/vehicle_aliases.json`：可扩展的品牌/车型别名配置。
- `.env.example`：环境变量模板。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 配置环境变量

在 `.env` 中填写：

- `FEISHU_APP_ID` / `FEISHU_APP_SECRET`：飞书开发者后台创建内部应用后获取。
- `FEISHU_TABLE_APP_TOKEN` / `FEISHU_TABLE_ID`：多维表格 URL 和表配置中获取。
- `FEISHU_CHAT_ID`：目标飞书群聊 ID，应用需要具备读取历史消息权限并加入群聊。
- `GLM_API_KEY`：智谱 GLM API Key，用于图片理解、结构化抽取和 embedding。
- `RELAY_API_KEY` / `RELAY_BASE_URL`：可选，OpenAI 兼容中转 API，用于图片低置信度兜底。

## 运行

无真实 key 时先跑 mock，验证表结构和接口层：

```bash
python etl.py --mock table
python etl.py --mock chat
```

处理飞书多维表格：

```bash
python etl.py table
```

处理飞书群聊历史：

```bash
python etl.py chat --start-time 1710000000000 --end-time 1712600000000
```

默认数据库写入 `data/knowledge.db`，图片写入 `data/images/`。

## 当前局限

- 向量检索存储在 SQLite 中，查询时逐条计算余弦相似度，适合第一阶段小规模数据，不适合大规模在线高并发。
- `configs/vehicle_aliases.json` 只内置常见示例，真实业务需要持续补充别名、年款和同代车型范围。
- 飞书图片 token 在不同来源中可能是 `file_token`、`image_key` 或附件 token；代码已做通用尝试，真实数据接入时需要用样本校验字段形态。
- 群聊切分是时间窗口加关键词/词重叠的轻量策略，足够做第一版离线清洗，但复杂多线程对话仍需要人工抽检。
- LLM 输出 JSON 依赖模型遵循提示词，生产化建议增加失败重试、人工审核队列和 schema 校验。
