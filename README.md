# 今晚到此

一个围绕“睡前最后 30 分钟”的 Python 本地 Web Demo。它用显式 Workflow 组织睡前收尾流程，用 SQLite 保存会话和用户偏好，用本地向量存储召回长期记忆和受控睡前建议，并在进入 App 时以较低音量播放用户设置的默认白噪音。用户还可以在低刺激过渡中勾选“还是睡不着？”，输入实时感受，获得下一轮建议；系统默认本地优先，也支持用户主动允许联网检索。源码目录只放内置资源；数据库、向量索引和用户上传音频默认放在 Windows 的 `%LOCALAPPDATA%\\TonightToHere`，也可以用 `TONIGHT_RUNTIME_DIR` 指定最终运行目录。

## 运行环境

- Python 3.10+
- Windows、macOS 或 Linux

## 快速启动

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

首次使用时，浏览器可能会阻止带声音的自动播放。页面会尝试自动播放默认白噪音；如果浏览器拦截，点击播放器一次即可获得授权。默认音量为 18%，可在侧边栏调整。结束今天后，音频可设置为 0 到 120 分钟后自动淡出；0 表示持续播放直到手动停止。

## 验证核心流程

不安装 Streamlit 也可以先运行无界面的核心流程测试：

```bash
python demo_cli.py
python -m unittest discover -s tests -v
```

默认使用确定性的本地兜底，因此没有 API Key 也可以运行完整流程。事件识别采用“模型语义分组优先、本地校验与失败回退”的方式：API 模式下模型直接阅读完整原文，判断省略连接词的隐含关系；本地逻辑不会再按逗号拆开模型已经合并的事件，只拆分带有“另外、还要、此外”等明确转向的事项。

## Streamlit Community Cloud 部署

仓库在不配置任何 Secrets 时会默认使用 `MODEL_MODE=mock`，适合公开 Demo。部署步骤：

1. 使用 GitHub 登录 [Streamlit Community Cloud](https://share.streamlit.io/)；
2. 选择本仓库和 `main` 分支；
3. Main file path 填写 `app.py`；
4. 点击 Deploy，等待平台生成固定的 `streamlit.app` 地址。

不要上传本地 `.env`、SQLite、向量索引或用户上传音频。Community Cloud 的本地文件系统可能在应用重启后重置，因此公开 Demo 不应依赖历史数据长期保存。

### 免费模型 API（推荐用于事件识别）

项目已兼容 OpenRouter 的 `openrouter/free` 免费模型路由。该路由会在当前可用的免费模型中自动选择；免费模型可能有速率、可用性和上下文限制。启用步骤：

1. 在 OpenRouter 创建免费 API Key；
2. 复制 `.env.example` 为 `.env`；
3. 将 `MODEL_MODE` 改为 `api`，并填写 `MODEL_API_KEY`；
4. 重启 Streamlit。

```text
MODEL_MODE=api
MODEL_API_KEY=你的_OpenRouter_Key
MODEL_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=openrouter/free
MODEL_TIMEOUT_SECONDS=45
```

API 不可用、限流、输出格式错误时会自动退回本地识别。睡前输入可能包含情绪、健康或性生活等敏感信息；只有用户明确同意将输入发送给第三方模型时，才应启用 `MODEL_MODE=api`。也可以把 `MODEL_BASE_URL`、`MODEL_NAME` 替换为其他 OpenAI 兼容接口。

如果密钥已经保存在 Windows 用户环境变量中，可以通过 `MODEL_API_KEY_ENV` 填写环境变量名称，项目只读取变量值，不需要把密钥写入 `.env`。例如火山方舟配置：

```text
MODEL_MODE=api
MODEL_API_KEY_ENV=HUOSHAN_FREE_API_KEY
MODEL_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
MODEL_NAME=你的推理接入点 ID
MODEL_MAX_TOKENS=800
MODEL_THINKING=disabled
```

当应用由受限账号、容器或 Codex 沙箱启动时，可用 `TONIGHT_RUNTIME_DIR` 指向明确可写且持久化的目录。SQLite、向量索引和用户上传音频会统一存放在该目录中。

## 可选 API 服务

```bash
uvicorn api:app --reload --port 8000
```

接口包括：

- `GET /health`
- `GET /audio`
- `GET /users/{user_id}/audio-preference`
- `PUT /users/{user_id}/audio-preference`
- `POST /audio/uploads`

## 睡不着反馈回合

低刺激动作建议先持续 5 到 10 分钟。用户勾选“还是睡不着？”后，可以输入当前感受或状态：

1. 召回本地审核资料和用户短期、长期上下文；
2. 用户明确允许时，对清洗后的主题词做一次联网搜索；
3. 清洗网页摘要、切分 chunk，并与本地 chunk 做 RRF 排序；
4. 交给 LLM 生成一条回应和一个低刺激行动；
5. 网络、检索或模型失败时，自动返回固定的通用建议。

每次会话最多两轮反馈，避免变成无限聊天。

## 项目结构

```text
app.py                 Streamlit 页面
api.py                 FastAPI 音频和偏好接口
advice.py              本地优先的建议编排和异常回退
search.py              可选网页搜索、清洗、切分和 RRF
workflow.py            显式状态流转
llm.py                 Mock/API 模型网关
memory.py              短期与长期记忆服务
context.py             LLM 上下文组装
rag.py                 受控普通 RAG
audio.py               内置与用户音频服务
database.py            SQLite 持久化
vector_store.py        本地向量存储适配器
data/knowledge         审核后的建议资料
tests                  核心流程测试
```
