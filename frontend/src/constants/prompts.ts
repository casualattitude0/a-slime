export type QuickPrompt = {
  label: string
  text: string
  systemInstruction?: string
}

export const QUICK_PROMPTS: QuickPrompt[] = [
  {
    label: '分析模式',
    systemInstruction: `任務目標：
- 你可以決定是否委派 Sub Agent。
- 你要專注蒐集「對問題有幫助」的資料。

執行規範：
1. 先在專案中使用 Grep/Shell 搜尋與問題相關的檔案與片段。
2. 把蒐集到的資料整理並存放到路徑：~/Developer/Agent/analysis_data
3. 將整理後資料交給 Sub Agent 分析。
4. 如有需要，Sub Agent 可以繼續委派更多子 Agent 做分工統整。`,
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
