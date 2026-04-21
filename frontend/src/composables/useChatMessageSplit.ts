import { computed, ref, watch, type Ref } from 'vue'
import type { Message } from '../stores/chatStore'

type TurnSelectionMode = 'latest' | 'selected'

export interface ChatTurn {
  start: number
  end: number
  messages: Message[]
}

export function useChatMessageSplit(messages: Ref<Message[]>) {
  const selectionMode = ref<TurnSelectionMode>('latest')
  const selectedTurnStart = ref<number | null>(null)

  const turns = computed<ChatTurn[]>(() => {
    const m = messages.value
    if (m.length === 0) return []

    const userIndices: number[] = []
    for (let i = 0; i < m.length; i++) {
      if (m[i]?.role === 'user') userIndices.push(i)
    }

    if (userIndices.length === 0) {
      return [{ start: 0, end: m.length - 1, messages: m.slice() }]
    }

    const out: ChatTurn[] = []
    for (let i = 0; i < userIndices.length; i++) {
      const start = userIndices[i]!
      const nextStart = userIndices[i + 1]
      const end = nextStart == null ? m.length - 1 : nextStart - 1
      out.push({ start, end, messages: m.slice(start, end + 1) })
    }
    return out
  })

  const newestTurnStart = computed(() => {
    const t = turns.value
    if (t.length === 0) return null
    return t[t.length - 1]!.start
  })

  const activeTurn = computed<ChatTurn | null>(() => {
    const t = turns.value
    if (t.length === 0) return null
    if (selectionMode.value === 'latest') return t[t.length - 1]!
    return t.find((turn) => turn.start === selectedTurnStart.value) ?? t[t.length - 1]!
  })

  const activeTurnIndex = computed(() => {
    const t = activeTurn.value
    if (!t) return -1
    return turns.value.findIndex((turn) => turn.start === t.start)
  })

  const historyTurns = computed<ChatTurn[]>(() => turns.value)

  const historyMessages = computed(() => {
    return historyTurns.value.flatMap((turn) => turn.messages)
  })

  const activeMessages = computed(() => {
    const t = activeTurn.value
    return t ? t.messages : ([] as Message[])
  })

  const activeStartIndex = computed(() => {
    const t = activeTurn.value
    return t ? t.start : messages.value.length
  })

  function globalIndexInActive(localIndex: number): number {
    const t = activeTurn.value
    if (!t) return -1
    return t.start + localIndex
  }

  function setActiveTurnByHistorySelection(turnStart: number) {
    selectionMode.value = 'selected'
    selectedTurnStart.value = turnStart
  }

  function focusNewestTurn() {
    selectionMode.value = 'latest'
    selectedTurnStart.value = newestTurnStart.value
  }

  watch(
    turns,
    (nextTurns) => {
      if (nextTurns.length === 0) {
        selectionMode.value = 'latest'
        selectedTurnStart.value = null
        return
      }
      if (selectionMode.value === 'selected') {
        const exists = nextTurns.some((turn) => turn.start === selectedTurnStart.value)
        if (!exists) {
          selectionMode.value = 'latest'
          selectedTurnStart.value = nextTurns[nextTurns.length - 1]!.start
        }
      }
    },
    { immediate: true }
  )

  return {
    turns,
    historyTurns,
    activeTurn,
    activeTurnIndex,
    historyMessages,
    activeMessages,
    activeStartIndex,
    globalIndexInActive,
    setActiveTurnByHistorySelection,
    focusNewestTurn,
  }
}
