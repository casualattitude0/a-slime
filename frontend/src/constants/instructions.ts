export type QuickInstruction = {
  label: string
  text: string
}

export const QUICK_PROMPTS: QuickInstruction[] = [
  {
    label: '新增行程到Apple行事曆',
    text: `請協助新增一筆行程到我的 Apple 行事曆（Calendar）。

請向我確認缺少的資訊，並在資訊齊全後提供可直接使用的做法（例如：行程標題、開始／結束時間（含時區）、地點、提醒、備註、是否重複）。

行程內容：
- 標題：
- 開始時間：
- 結束時間：
- 時區： Asia/Taipei
- 地點：（可選）
- 提醒：（例如 行程前 15 分鐘）
- 備註：（可選）`,
  },
]
