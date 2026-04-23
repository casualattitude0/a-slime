export type QuickPrompt = {
  label: string
  text: string
}

export const QUICK_PROMPTS: QuickPrompt[] = [
  {
    label: '新增事件到Google行事曆',
    text: `行程內容：
- 標題：
- 開始時間：
- 結束時間：
- 時區： Asia/Taipei
- 地點：（可選）
- 提醒：行程前 60 分鐘
- 備註：（可選）

需求：
- 只新增到 Google Calendar，不要新增到 Apple Calendar`,
  },
  {
    label: '新增事件到Google+Apple',
    text: `行程內容：
- 標題：
- 開始時間：
- 結束時間：
- 時區： Asia/Taipei
- 地點：（可選）
- 提醒：行程前 60 分鐘
- 備註：（可選）

需求：
- 同步新增到 Google Calendar 與 Apple Calendar`,
  },
  {
    label: '新增行程到Apple行事曆',
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
