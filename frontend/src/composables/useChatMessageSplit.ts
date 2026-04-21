import { computed, type Ref } from 'vue'
import type { Message } from '../stores/chatStore'

export function useChatMessageSplit(messages: Ref<Message[]>) {
  const lastUserIndex = computed(() => {
    const m = messages.value
    for (let i = m.length - 1; i >= 0; i--) {
      if (m[i]?.role === 'user') return i
    }
    return -1
  })

  const historyMessages = computed(() => {
    const i = lastUserIndex.value
    if (i <= 0) return [] as Message[]
    return messages.value.slice(0, i)
  })

  const activeMessages = computed(() => {
    const i = lastUserIndex.value
    if (i < 0) return messages.value
    return messages.value.slice(i)
  })

  const activeStartIndex = computed(() => (lastUserIndex.value < 0 ? 0 : lastUserIndex.value))

  function globalIndexInActive(localIndex: number): number {
    const i = lastUserIndex.value
    if (i < 0) return localIndex
    return i + localIndex
  }

  return {
    lastUserIndex,
    historyMessages,
    activeMessages,
    activeStartIndex,
    globalIndexInActive,
  }
}
