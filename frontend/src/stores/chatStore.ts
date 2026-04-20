import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface LLMErrorPayload {
  is_llm_error: boolean
  error_type: string
  message: string
  retry_after_seconds: number | null
}

export interface Message {
  role: 'user' | 'bot' | 'err' | 'thought'
  text: string
  llmError?: LLMErrorPayload
}

export interface VersionEntry {
  version_id: string
  name: string
  session_id: string
  memory_collection: string
  rag_collection: string
  model_profile: string
  created_at: string
  is_active: boolean
}

export interface MemoryItem {
  id: string
  content: string
  metadata: Record<string, any>
}

export interface RagItem {
  id: string
  content: string
  metadata: Record<string, any>
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const sessionId = ref<string | null>(localStorage.getItem('agent_session_id'))
  const status = ref<string>('')
  const isLoading = ref<boolean>(false)
  const pendingLLMError = ref<{ messageIndex: number; payload: LLMErrorPayload; originalText: string } | null>(null)

  const versions = ref<VersionEntry[]>([])
  const activeVersionId = ref<string | null>(null)
  const availableProfiles = ref<string[]>(['default'])

  const memoryItems = ref<MemoryItem[]>([])
  const memoryLoading = ref<boolean>(false)

  const ragItems = ref<RagItem[]>([])
  const ragLoading = ref<boolean>(false)
  const activeController = ref<AbortController | null>(null)

  // Index of the bot message currently being streamed (-1 = none).
  const streamingBotIndex = ref<number>(-1)
  // Preferred transport: 'sse' | 'ws'
  const transport = ref<'sse' | 'ws'>('ws')
  // Active WebSocket for the current request (if ws transport).
  let _activeWs: WebSocket | null = null

  function setSessionId(id: string | null) {
    sessionId.value = id
    if (id) {
      localStorage.setItem('agent_session_id', id)
    } else {
      localStorage.removeItem('agent_session_id')
    }
  }

  /** Handle one parsed event frame from either SSE or WebSocket transport. */
  function _handleStreamEvent(obj: any, originalText: string): { done: boolean } {
    if (obj.event === 'start') {
      if (obj.session_id) setSessionId(obj.session_id)
    }

    if (obj.event === 'status' && obj.label) {
      status.value = String(obj.label)
      // Insert thought bubbles only while the bot hasn't started replying yet.
      if (streamingBotIndex.value === -1) {
        messages.value.push({ role: 'thought', text: String(obj.label) })
      }
    }

    if (obj.event === 'delta' && obj.text) {
      if (streamingBotIndex.value === -1) {
        // First delta: push a live bot message placeholder.
        streamingBotIndex.value = messages.value.length
        messages.value.push({ role: 'bot', text: String(obj.text) })
      } else {
        messages.value[streamingBotIndex.value].text += String(obj.text)
      }
    }

    if (obj.event === 'done') {
      if (obj.session_id) setSessionId(obj.session_id)
      status.value = ''

      if (obj.terminated) {
        streamingBotIndex.value = -1
        return { done: true }
      }

      if (obj.error) {
        // Remove any partial bot message that was being built.
        if (streamingBotIndex.value !== -1) {
          messages.value.splice(streamingBotIndex.value, 1)
          streamingBotIndex.value = -1
        }
        const llmErr: LLMErrorPayload | undefined = obj.llm_error ?? undefined
        const idx = messages.value.length
        messages.value.push({ role: 'err', text: obj.error, llmError: llmErr })
        if (llmErr?.is_llm_error) {
          pendingLLMError.value = { messageIndex: idx, payload: llmErr, originalText }
        }
      } else if (obj.reply) {
        if (streamingBotIndex.value !== -1) {
          // Replace the streaming placeholder with the authoritative final text.
          messages.value[streamingBotIndex.value].text = obj.reply
        } else {
          messages.value.push({ role: 'bot', text: obj.reply })
        }
      }
      streamingBotIndex.value = -1
      return { done: true }
    }

    if (obj.event === 'error') {
      status.value = ''
      if (streamingBotIndex.value !== -1) {
        messages.value.splice(streamingBotIndex.value, 1)
        streamingBotIndex.value = -1
      }
      messages.value.push({ role: 'err', text: obj.error || 'Stream error' })
      return { done: true }
    }

    return { done: false }
  }

  async function _sendSSE(text: string) {
    activeController.value = new AbortController()
    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: activeController.value.signal,
        body: JSON.stringify({ message: text, session_id: sessionId.value }),
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        const d = data.detail
        const errText = Array.isArray(d)
          ? d.map((x: any) => x.msg || JSON.stringify(x)).join('; ')
          : (d || res.statusText || 'Request failed')
        messages.value.push({ role: 'err', text: errText })
        return
      }

      const reader = res.body?.getReader()
      if (!reader) {
        messages.value.push({ role: 'err', text: 'No response body' })
        return
      }

      const dec = new TextDecoder()
      let buf = ''

      outer: while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        let nl
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl).trim()
          buf = buf.slice(nl + 1)
          if (!line) continue
          let obj: any
          try { obj = JSON.parse(line) } catch { continue }
          const { done: streamDone } = _handleStreamEvent(obj, text)
          if (streamDone) break outer
        }
      }
    } catch (e: any) {
      if (e?.name === 'AbortError') { status.value = ''; return }
      status.value = ''
      if (streamingBotIndex.value !== -1) {
        messages.value.splice(streamingBotIndex.value, 1)
        streamingBotIndex.value = -1
      }
      messages.value.push({ role: 'err', text: String(e) })
    } finally {
      activeController.value = null
    }
  }

  async function _sendWS(text: string) {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${location.host}/ws/chat/live`

    await new Promise<void>((resolve) => {
      const ws = new WebSocket(wsUrl)
      _activeWs = ws

      ws.onopen = () => {
        ws.send(JSON.stringify({ message: text, session_id: sessionId.value }))
      }

      ws.onmessage = (ev) => {
        let obj: any
        try { obj = JSON.parse(ev.data) } catch { return }
        const { done: streamDone } = _handleStreamEvent(obj, text)
        if (streamDone) ws.close()
      }

      ws.onerror = () => {
        status.value = ''
        if (streamingBotIndex.value !== -1) {
          messages.value.splice(streamingBotIndex.value, 1)
          streamingBotIndex.value = -1
        }
        messages.value.push({ role: 'err', text: 'WebSocket error' })
        resolve()
      }

      ws.onclose = () => {
        _activeWs = null
        resolve()
      }
    })
  }

  async function sendMessage(text: string) {
    if (!text.trim() || isLoading.value) return

    messages.value.push({ role: 'user', text })
    isLoading.value = true
    status.value = ''
    pendingLLMError.value = null
    streamingBotIndex.value = -1

    try {
      if (transport.value === 'ws' && typeof WebSocket !== 'undefined') {
        await _sendWS(text)
      } else {
        await _sendSSE(text)
      }
    } finally {
      streamingBotIndex.value = -1
      status.value = ''
      isLoading.value = false
    }
  }

  async function terminateMessage() {
    if (!isLoading.value) return
    status.value = ''
    const sid = sessionId.value
    activeController.value?.abort()
    activeController.value = null
    if (_activeWs) {
      _activeWs.close()
      _activeWs = null
    }
    streamingBotIndex.value = -1
    isLoading.value = false
    try {
      if (sid) {
        void fetch('/api/chat/terminate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sid }),
        })
      }
    } catch {
    }
  }

  async function clearHistory() {
    const sid = sessionId.value
    isLoading.value = true
    status.value = ''
    try {
      await fetch('/api/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid }),
      })
      messages.value = []
    } catch (e: any) {
      messages.value.push({ role: 'err', text: String(e) })
    } finally {
      isLoading.value = false
    }
  }

  function fixIssue() {
    if (!pendingLLMError.value) return
    const { payload } = pendingLLMError.value
    const lines: string[] = ['**LLM 發生錯誤，請依下列步驟處理：**']
    if (payload.error_type === 'quota_exceeded') {
      lines.push('1. 確認 API 配額是否已用完（前往 https://ai.dev/rate-limit 查看用量）')
      lines.push('2. 確認帳單是否有效（https://ai.google.dev/gemini-api/docs/rate-limits）')
      lines.push('3. 若使用免費層，可等待配額重置後重試')
    } else if (payload.error_type === 'timeout' || payload.error_type === 'service_unavailable') {
      lines.push('1. 服務暫時不可用，請稍後重試')
      lines.push('2. 確認網路連線是否正常')
    } else {
      lines.push('1. 檢查 GOOGLE_API_KEY / GEMINI_API_KEY 是否正確設定')
      lines.push('2. 確認模型名稱與可用區域')
    }
    if (payload.retry_after_seconds) {
      lines.push(`4. 建議等待 ${payload.retry_after_seconds} 秒後再試`)
    }
    messages.value.push({ role: 'bot', text: lines.join('\n') })
    pendingLLMError.value = null
  }

  async function answerImmediately() {
    const pending = pendingLLMError.value
    if (!pending || !sessionId.value) return
    pendingLLMError.value = null
    isLoading.value = true
    status.value = ''
    try {
      const res = await fetch('/api/chat/fallback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: pending.originalText, session_id: sessionId.value }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        const d = data.detail
        const errText = Array.isArray(d)
          ? d.map((x: any) => x.msg || JSON.stringify(x)).join('; ')
          : (d || res.statusText || 'Request failed')
        messages.value.push({ role: 'err', text: errText })
      } else if (data.error) {
        messages.value.push({ role: 'err', text: data.error })
      } else if (data.reply) {
        messages.value.push({ role: 'bot', text: data.reply })
      }
    } catch (e: any) {
      messages.value.push({ role: 'err', text: String(e) })
    } finally {
      isLoading.value = false
      status.value = ''
    }
  }

  // ── Version actions ──────────────────────────────────────────────────────

  async function fetchVersions() {
    try {
      const res = await fetch('/api/versions')
      if (!res.ok) return
      const data = await res.json()
      versions.value = data.versions ?? []
      activeVersionId.value = data.active_version_id ?? null
      availableProfiles.value = data.available_model_profiles ?? ['default']
    } catch {}
  }

  async function switchVersion(versionId: string): Promise<boolean> {
    try {
      const res = await fetch('/api/versions/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ version_id: versionId }),
      })
      if (!res.ok) return false
      const data = await res.json()
      if (data.session_id) setSessionId(data.session_id)
      messages.value = []
      await fetchVersions()
      return true
    } catch {
      return false
    }
  }

  async function createVersion(name: string, modelProfile: string = 'default'): Promise<boolean> {
    try {
      const res = await fetch('/api/versions/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, model_profile: modelProfile }),
      })
      if (!res.ok) return false
      await fetchVersions()
      return true
    } catch {
      return false
    }
  }

  async function deleteVersion(versionId: string): Promise<boolean> {
    try {
      const res = await fetch(`/api/versions/${versionId}`, { method: 'DELETE' })
      if (!res.ok) return false
      const data = await res.json()
      if (versionId === activeVersionId.value) {
        messages.value = []
        setSessionId(null)
      }
      activeVersionId.value = data.active_version_id ?? null
      await fetchVersions()
      return true
    } catch {
      return false
    }
  }

  // ── Memory actions ───────────────────────────────────────────────────────

  async function fetchMemoryItems() {
    memoryLoading.value = true
    try {
      const res = await fetch('/api/memory/items')
      if (!res.ok) return
      const data = await res.json()
      memoryItems.value = data.items ?? []
    } catch {
    } finally {
      memoryLoading.value = false
    }
  }

  async function deleteMemoryItem(itemId: string): Promise<boolean> {
    try {
      const res = await fetch(`/api/memory/items/${encodeURIComponent(itemId)}`, {
        method: 'DELETE',
      })
      if (!res.ok) return false
      memoryItems.value = memoryItems.value.filter((m) => m.id !== itemId)
      return true
    } catch {
      return false
    }
  }

  async function deleteAllMemory(): Promise<boolean> {
    try {
      const res = await fetch('/api/memory/all', { method: 'DELETE' })
      if (!res.ok) return false
      memoryItems.value = []
      return true
    } catch {
      return false
    }
  }

  // ── RAG actions ──────────────────────────────────────────────────────────

  async function fetchRagItems() {
    ragLoading.value = true
    try {
      const res = await fetch('/api/rag/items')
      if (!res.ok) return
      const data = await res.json()
      ragItems.value = data.items ?? []
    } catch {
    } finally {
      ragLoading.value = false
    }
  }

  async function deleteRagItem(itemId: string): Promise<boolean> {
    try {
      const res = await fetch(`/api/rag/items/${encodeURIComponent(itemId)}`, {
        method: 'DELETE',
      })
      if (!res.ok) return false
      ragItems.value = ragItems.value.filter((r) => r.id !== itemId)
      return true
    } catch {
      return false
    }
  }

  async function deleteAllRag(): Promise<boolean> {
    try {
      const res = await fetch('/api/rag/all', { method: 'DELETE' })
      if (!res.ok) return false
      ragItems.value = []
      return true
    } catch {
      return false
    }
  }

  // ── Global nuke ──────────────────────────────────────────────────────────

  async function deleteAllData(): Promise<boolean> {
    try {
      const res = await fetch('/api/data/all', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm_token: 'DELETE_ALL' }),
      })
      if (!res.ok) return false
      messages.value = []
      memoryItems.value = []
      ragItems.value = []
      setSessionId(null)
      await fetchVersions()
      return true
    } catch {
      return false
    }
  }

  return {
    messages,
    sessionId,
    status,
    isLoading,
    pendingLLMError,
    streamingBotIndex,
    transport,
    versions,
    activeVersionId,
    availableProfiles,
    memoryItems,
    memoryLoading,
    ragItems,
    ragLoading,
    sendMessage,
    terminateMessage,
    clearHistory,
    fixIssue,
    answerImmediately,
    fetchVersions,
    switchVersion,
    createVersion,
    deleteVersion,
    fetchMemoryItems,
    deleteMemoryItem,
    deleteAllMemory,
    fetchRagItems,
    deleteRagItem,
    deleteAllRag,
    deleteAllData,
  }
})
