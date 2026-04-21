<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'
import { SendHorizontal, Square } from 'lucide-vue-next'

const props = defineProps<{
  disabled: boolean
  loading: boolean
}>()

const emit = defineEmits<{
  (e: 'send', payload: { text: string; llmMode: 'auto' | 'gemini' | 'agent' }): void
  (e: 'terminate'): void
}>()

const input = ref('')
const llmMode = ref<'auto' | 'gemini' | 'agent'>('auto')
const textareaRef = ref<HTMLTextAreaElement | null>(null)
const isComposingWithIME = ref(false)

const adjustHeight = () => {
  if (!textareaRef.value) return
  textareaRef.value.style.height = 'auto'
  textareaRef.value.style.height = `${Math.min(textareaRef.value.scrollHeight, 200)}px`
}

const handleKeydown = (e: KeyboardEvent) => {
  const stillComposing = isComposingWithIME.value || e.isComposing || e.keyCode === 229
  const isCmdEnter = e.key === 'Enter' && e.metaKey && !e.ctrlKey && !e.altKey && !e.shiftKey
  if (isCmdEnter && !stillComposing) {
    e.preventDefault()
    send()
  }
}

const handleCompositionStart = () => { isComposingWithIME.value = true }
const handleCompositionEnd = () => { isComposingWithIME.value = false }

const send = () => {
  const text = input.value.trim()
  if (!text || props.disabled) return
  emit('send', { text, llmMode: llmMode.value })
  input.value = ''
  nextTick(() => {
    adjustHeight()
    textareaRef.value?.focus()
  })
}

const terminate = () => {
  if (!props.loading) return
  emit('terminate')
}

onMounted(() => {
  textareaRef.value?.focus()
})
</script>

<template>
  <div class="composer" :class="{ 'composer--disabled': disabled }">
    <select v-model="llmMode" class="llm-select" :disabled="disabled || loading" aria-label="LLM mode">
      <option value="auto">Auto</option>
      <option value="gemini">Gemini</option>
      <option value="agent">Agent</option>
    </select>
    <textarea
      ref="textareaRef"
      v-model="input"
      class="composer-input"
      placeholder="Message agent…"
      :disabled="disabled"
      @input="adjustHeight"
      @keydown="handleKeydown"
      @compositionstart="handleCompositionStart"
      @compositionend="handleCompositionEnd"
      rows="1"
    ></textarea>

    <div class="composer-actions">
      <span v-if="!loading" class="send-hint">⌘↵</span>
      <button
        @click="loading ? terminate() : send()"
        :disabled="loading ? false : (!input.trim() || disabled)"
        class="send-btn"
        :class="{ 'send-btn--ready': !loading && input.trim() && !disabled, 'send-btn--terminate': loading }"
        :title="loading ? 'Terminate' : 'Send (⌘↵)'"
      >
        <Square v-if="loading" :size="14" :fill="'currentColor'" />
        <SendHorizontal v-else :size="15" />
      </button>
    </div>
  </div>
</template>

<style scoped>
.composer {
  display: flex;
  align-items: flex-end;
  gap: 10px;
  padding: 10px 12px 10px 16px;
  background: var(--surface);
  border: 1px solid var(--border-bright);
  border-radius: 13px;
  transition: border-color 0.18s ease, box-shadow 0.18s ease;
}

.composer:focus-within {
  border-color: rgba(var(--accent-rgb), 0.5);
  box-shadow:
    0 0 0 1px rgba(var(--accent-rgb), 0.2),
    0 0 24px rgba(var(--accent-rgb), 0.18);
}

.composer--disabled {
  opacity: 0.85;
}

.composer-input {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  resize: none;
  color: var(--text);
  font-family: inherit;
  font-size: 14px;
  line-height: 1.55;
  min-height: 22px;
  max-height: 200px;
  overflow-y: auto;
  padding: 0;
}

.composer-input::placeholder {
  color: var(--text-dim);
}

.llm-select {
  background: var(--surface-2);
  border: 1px solid var(--border-bright);
  color: var(--text-dim);
  border-radius: 8px;
  font-size: 12px;
  height: 32px;
  padding: 0 8px;
  outline: none;
  flex-shrink: 0;
}

.llm-select:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.composer-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
  padding-bottom: 1px;
}

.send-hint {
  font-size: 10px;
  color: var(--text-dim);
  font-family: ui-monospace, monospace;
  opacity: 0.5;
  user-select: none;
}

.send-btn {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--surface-2);
  border: 1px solid var(--border-bright);
  color: var(--text-dim);
  cursor: pointer;
  transition: all 0.15s ease;
  flex-shrink: 0;
}

.send-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.send-btn--ready {
  background: var(--accent);
  border-color: var(--accent);
  color: #0a0b0f;
  box-shadow: 0 0 10px var(--accent-glow);
}

.send-btn--ready:hover {
  box-shadow: 0 0 18px var(--accent-glow);
  transform: scale(1.06);
}

.send-btn--terminate {
  background: var(--error-soft);
  border-color: rgba(var(--error-rgb), 0.42);
  color: var(--error);
}
</style>
