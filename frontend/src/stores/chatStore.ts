import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface Message {
  role: 'user' | 'bot' | 'err'
  text: string
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

  const versions = ref<VersionEntry[]>([])
  const activeVersionId = ref<string | null>(null)
  const availableProfiles = ref<string[]>(['default'])

  const memoryItems = ref<MemoryItem[]>([])
  const memoryLoading = ref<boolean>(false)

  const ragItems = ref<RagItem[]>([])
  const ragLoading = ref<boolean>(false)

  function setSessionId(id: string | null) {
    sessionId.value = id
    if (id) {
      localStorage.setItem('agent_session_id', id)
    } else {
      localStorage.removeItem('agent_session_id')
    }
  }

  async function sendMessage(text: string) {
    if (!text.trim() || isLoading.value) return

    messages.value.push({ role: 'user', text })
    isLoading.value = true
    status.value = ''

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          session_id: sessionId.value,
        }),
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        const d = data.detail
        const errText = Array.isArray(d)
          ? d.map((x: any) => x.msg || JSON.stringify(x)).join('; ')
          : (d || res.statusText || 'Request failed')
        messages.value.push({ role: 'err', text: errText })
        isLoading.value = false
        return
      }

      const reader = res.body?.getReader()
      if (!reader) {
        messages.value.push({ role: 'err', text: 'No response body' })
        isLoading.value = false
        return
      }

      const dec = new TextDecoder()
      let buf = ''
      let finalDone: any = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        let nl
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl).trim()
          buf = buf.slice(nl + 1)
          if (!line) continue
          let obj
          try {
            obj = JSON.parse(line)
          } catch {
            continue
          }
          if (obj.event === 'status' && obj.label) {
            status.value = String(obj.label)
          }
          if (obj.event === 'done') {
            finalDone = obj
          }
        }
      }

      if (finalDone && finalDone.session_id) {
        setSessionId(finalDone.session_id)
      }
      status.value = ''

      if (finalDone && finalDone.error) {
        messages.value.push({ role: 'err', text: finalDone.error })
      }
      if (finalDone && finalDone.reply) {
        messages.value.push({ role: 'bot', text: finalDone.reply })
      }
    } catch (e: any) {
      status.value = ''
      messages.value.push({ role: 'err', text: String(e) })
    } finally {
      isLoading.value = false
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
    versions,
    activeVersionId,
    availableProfiles,
    memoryItems,
    memoryLoading,
    ragItems,
    ragLoading,
    sendMessage,
    clearHistory,
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
