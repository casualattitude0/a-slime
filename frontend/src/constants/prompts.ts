export type QuickPrompt = {
  label: string
  text: string
  systemInstruction?: string
}

export const QUICK_PROMPTS: QuickPrompt[] = [
  {
    label: '分析模式',
    systemInstruction: `[ANALYSIS_MODE_ACTIVE]
禁止向使用者詢問是否有資料或文件。直接依以下步驟自主執行。

任務目標：
- 你可以決定是否委派 Sub Agent。
- 你要專注蒐集「對問題有幫助」的資料。

參考資料：
- 所有參考資料統一放在 ~/Developer/Agent/analysis_data/reference。
- 執行前必須先讀取該路徑下相關檔案，作為分析的輸入來源。

執行規範：
1. 使用 execute_shell_command 讀取 ~/Developer/Agent/analysis_data/reference 下的相關檔案。
2. 把蒐集到的資料整理後交給 Sub Agent 分析。
3. 如有需要，Sub Agent 可以繼續委派更多子 Agent 做分工統整。

輸出規範：
- 分析完成後，必須使用 export_document tool 將結果輸出到 ~/Developer/Agent/analysis_data/export。
- 輸出格式為 Markdown（.md），檔名應描述分析主題，例如 analysis_report.md。
- 輸出內容結構：
  ## 分析主題
  ## 資料來源
  ## 分析結果
  ## 結論與建議`,
    text: '',
  },
  {
    label: '新增事件到Google行事曆',
    systemInstruction: `只新增到 Google Calendar，不要新增到 Apple Calendar。`,
    text: `行程內容：
- 標題：
- 開始時間：
- 結束時間：
- 時區： Asia/Taipei
- 地點：（可選）
- 提醒：行程前 60 分鐘
- 備註：（可選）`,
  },
  {
    label: '新增事件到Google+Apple',
    systemInstruction: `同步新增到 Google Calendar 與 Apple Calendar。`,
    text: `行程內容：
- 標題：
- 開始時間：
- 結束時間：
- 時區： Asia/Taipei
- 地點：（可選）
- 提醒：行程前 60 分鐘
- 備註：（可選）`,
  },
  {
    label: '新增行程到Apple行事曆',
    systemInstruction: `只新增到 Apple Calendar，不要新增到 Google Calendar。`,
    text: `行程內容：
- 標題：
- 開始時間：
- 結束時間：
- 時區： Asia/Taipei
- 地點：（可選）
- 提醒：行程前 60 分鐘
- 備註：（可選）`,
  },
]
