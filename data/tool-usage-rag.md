# Agent 工具使用說明（RAG）

## 目標

本文件提供 Agent 可用工具的用途、何時使用、輸入格式與簡短範例，供檢索與回答時引用。

## 工具總覽

### 1) search_memory

- 用途：搜尋長期記憶中的既有事實、偏好、歷史決策。
- 何時使用：問題可能依賴過往對話或已保存資訊時，優先查詢。
- 輸入：
  - `query`：要查的問題或關鍵字
  - `k`：最多回傳筆數（1-10）
- 範例：
  - query: `使用者偏好的回覆語言`
  - k: `5`

### 2) save_to_memory

- 用途：把可重用的重要資訊存入長期記憶。
- 何時使用：使用者偏好、固定規則、重要決策、可復用結論。
- 輸入：
  - `content`：要保存的內容
  - `tags`：可選，逗號分隔
- 範例：
  - content: `使用者偏好繁體中文，回覆需精簡`
  - tags: `preference,language`

### 3) document_search

- 用途：搜尋已匯入向量資料庫的本機文件內容。
- 何時使用：詢問 data 目錄文件內容、內部知識、專案文檔時。
- 輸入：
  - `query`：問題或關鍵字
- 範例：
  - query: `高優先級任務有哪些`

### 4) web_search

- 用途：搜尋公開網路的最新資訊。
- 何時使用：需要即時外部資訊，且本機文件/記憶不足時。
- 輸入：
  - `query`：搜尋關鍵字
  - `max_results`：最多結果數（1-10）
- 範例：
  - query: `Python 3.13 release notes`
  - max_results: `5`

### 5) web_fetch

- 用途：抓取指定網址並抽出可閱讀文字。
- 何時使用：已知道候選網址，需讀取內文時。
- 輸入：
  - `url`：完整 http(s) 網址
- 範例：
  - url: `https://docs.python.org/3/whatsnew/3.13.html`

### 6) execute_shell_command

- 用途：執行本機 Shell 指令取得系統資料、檔案片段或時間資訊。
- 何時使用：需要 `date`、`grep`、`tail`、`ls` 等本機命令結果時。
- 輸入：
  - `command`：要執行的 shell 指令
- 範例：
  - command: `date`
  - command: `tail -n 50 logs/app.log`
  - command: `grep -n "ERROR" logs/app.log`
- 注意：
  - 指令有逾時限制。
  - 輸出可能被截斷。
  - 回傳會包含 `exit_code`、`stdout`、`stderr`。

### 7) ask_reasoning_model

- 用途：把複雜推理或多步驟綜整委派給較強模型。
- 何時使用：跨來源比對、複雜判斷、長脈絡整合。
- 輸入：
  - `question`：要解的複雜問題
  - `context`：已蒐集背景（可選）
- 範例：
  - question: `比較兩個方案的風險與成本`
  - context: `方案A/B限制、預算、時程`

## 建議工具流程

1. 先查 `search_memory`，避免重做已知結論。
2. 若是本機知識，查 `document_search`。
3. 若需最新資訊，走 `web_search` + `web_fetch`。
4. 若需本機即時資料（例如今天日期、log 片段），用 `execute_shell_command`。
5. 需要深度判斷時，再用 `ask_reasoning_model`。
6. 可復用結論以 `save_to_memory` 收斂。

## 即時資訊規則

- 問「今天日期 / 現在時間 / 幾點」時，優先使用 `execute_shell_command` 執行 `date`，不可憑記憶回答。