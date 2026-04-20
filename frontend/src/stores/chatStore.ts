import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface Message {
  role: 'user' | 'bot' | 'err'
  text: string
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const sessionId = ref<string | null>(localStorage.getItem('agent_session_id'))
  const status = ref<string>('')
  const isLoading = ref<boolean>(false)

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

  return {
    messages,
    sessionId,
    status,
    isLoading,
    sendMessage,
    clearHistory,
  }
})
