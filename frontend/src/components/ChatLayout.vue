<script setup lang="ts">
import { ref, watch, nextTick, onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { Trash2, Loader2, Database, Zap } from 'lucide-vue-next'
import { useChatStore } from '../stores/chatStore'
import ChatMessage from './ChatMessage.vue'
import ChatInput from './ChatInput.vue'
import MemoryPanel from './MemoryPanel.vue'

const chatStore = useChatStore()
const { messages, status, isLoading, activeVersionId, versions, pendingLLMError } = storeToRefs(chatStore)
const logRef = ref<HTMLElement | null>(null)
const showPanel = ref(false)

const scrollToBottom = async () => {
  await nextTick()
  if (logRef.value) {
    logRef.value.scrollTop = logRef.value.scrollHeight
  }
}

watch(
  () => messages.value,
  () => { scrollToBottom() },
  { deep: true }
)

onMounted(async () => {
  scrollToBottom()
  await chatStore.fetchVersions()
})

const handleSend = (text: string) => {
  chatStore.sendMessage(text)
}

const handleTerminate = () => {
  chatStore.terminateMessage()
}

const handleClear = () => {
  if (confirm('Clear all chat history?')) {
    chatStore.clearHistory()
  }
}

const activeVersionName = () => {
  const v = versions.value.find((v) => v.version_id === activeVersionId.value)
  return v?.name ?? '—'
}
</script>

<template>
  <div class="root-layout">
    <!-- Command Bar -->
    <header class="command-bar">
      <div class="flex items-center gap-3">
        <div class="flex items-center gap-2">
          <span class="status-dot" :class="isLoading ? 'dot-active' : ''"></span>
          <span class="agent-name">LOCAL AGENT</span>
        </div>
        <span class="version-badge">{{ activeVersionName() }}</span>
      </div>

      <div class="flex items-center gap-1">
        <button
          @click="showPanel = !showPanel"
          class="cmd-btn"
          :class="showPanel ? 'cmd-btn--active' : ''"
          title="RAG / Memory"
        >
          <Database :size="14" />
          <span class="cmd-btn-label">Memory</span>
        </button>
        <button
          @click="handleClear"
          class="cmd-btn cmd-btn--danger"
          :disabled="isLoading"
          title="Clear history"
        >
          <Trash2 :size="14" />
        </button>
      </div>
    </header>

    <!-- Body: chat canvas + optional panel -->
    <div class="body-row">
      <!-- Message Canvas -->
      <main ref="logRef" class="chat-canvas">
        <!-- Empty state -->
        <div v-if="messages.length === 0" class="empty-state">
          <div class="empty-icon">
            <Zap :size="26" />
          </div>
          <p class="empty-title">Ready</p>
          <p class="empty-sub">Send a message to start</p>
        </div>

        <div v-else class="messages-inner">
        <ChatMessage
          v-for="(msg, i) in messages"
          :key="i"
          :role="msg.role"
          :text="msg.text"
          :llm-error="msg.llmError"
          :show-actions="pendingLLMError?.messageIndex === i"
          @fix-issue="chatStore.fixIssue()"
          @answer-immediately="chatStore.answerImmediately()"
        />
        </div>
      </main>

      <!-- Memory Panel -->
      <Transition name="panel">
        <div v-if="showPanel" class="panel-wrapper">
          <MemoryPanel />
        </div>
      </Transition>
    </div>

    <!-- Composer Footer -->
    <footer class="composer-footer">
      <div class="composer-inner">
        <!-- Status strip -->
        <div class="status-strip" :class="(status || isLoading) ? '' : 'status-strip--hidden'">
          <Loader2 :size="11" class="spin-icon" />
          <span>{{ status || 'Thinking…' }}</span>
        </div>
        <ChatInput :disabled="isLoading" :loading="isLoading" @send="handleSend" @terminate="handleTerminate" />
      </div>
    </footer>
  </div>
</template>

<style scoped>
.root-layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
  background: var(--bg);
}

/* ── Command bar ─────────────────────────────────────── */
.command-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 44px;
  flex-shrink: 0;
  background: rgba(17, 19, 24, 0.92);
  backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
  position: relative;
  z-index: 20;
}

.command-bar::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(0, 229, 255, 0.15) 30%, rgba(0, 229, 255, 0.15) 70%, transparent 100%);
}

.agent-name {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: var(--text);
  font-family: ui-monospace, monospace;
}

.version-badge {
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 20px;
  border: 1px solid rgba(0, 229, 255, 0.18);
  color: rgba(0, 229, 255, 0.55);
  font-family: ui-monospace, monospace;
  letter-spacing: 0.04em;
  background: rgba(0, 229, 255, 0.04);
}

/* status dot */
.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: rgba(0, 229, 255, 0.35);
  transition: background 0.3s ease, box-shadow 0.3s ease;
}

.status-dot.dot-active {
  background: var(--accent);
  box-shadow: 0 0 8px var(--accent-glow);
  animation: dot-pulse 1.4s ease-in-out infinite;
}

@keyframes dot-pulse {
  0%, 100% { box-shadow: 0 0 6px var(--accent-glow); }
  50% { box-shadow: 0 0 14px var(--accent-glow), 0 0 4px var(--accent); }
}

/* command buttons */
.cmd-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 12px;
  color: var(--text-dim);
  background: transparent;
  border: 1px solid transparent;
  cursor: pointer;
  transition: all 0.14s ease;
  white-space: nowrap;
}

.cmd-btn:hover {
  color: var(--text);
  background: var(--surface-2);
  border-color: var(--border-bright);
}

.cmd-btn-label {
  display: none;
}

@media (min-width: 540px) {
  .cmd-btn-label {
    display: inline;
  }
}

.cmd-btn--active {
  color: var(--accent);
  background: var(--accent-soft);
  border-color: rgba(0, 229, 255, 0.22);
}

.cmd-btn--danger:hover {
  color: var(--error);
  background: var(--error-soft);
  border-color: rgba(255, 77, 106, 0.22);
}

.cmd-btn:disabled {
  opacity: 0.38;
  cursor: not-allowed;
}

/* ── Body ────────────────────────────────────────────── */
.body-row {
  display: flex;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

/* ── Chat canvas ─────────────────────────────────────── */
.chat-canvas {
  flex: 1;
  overflow-y: auto;
  padding: 24px 20px 12px;
  scroll-behavior: smooth;
}

.messages-inner {
  max-width: 760px;
  margin: 0 auto;
}

/* Empty state */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: 10px;
  color: var(--text-dim);
}

.empty-icon {
  width: 54px;
  height: 54px;
  border-radius: 14px;
  border: 1px solid rgba(0, 229, 255, 0.18);
  background: rgba(0, 229, 255, 0.04);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--accent);
  margin-bottom: 4px;
}

.empty-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
  margin: 0;
}

.empty-sub {
  font-size: 12px;
  color: var(--text-dim);
  margin: 0;
}

/* ── Memory panel ────────────────────────────────────── */
.panel-wrapper {
  width: 288px;
  flex-shrink: 0;
  border-left: 1px solid var(--border);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* ── Composer footer ─────────────────────────────────── */
.composer-footer {
  flex-shrink: 0;
  padding: 8px 20px 20px;
  background: linear-gradient(to top, var(--bg) 55%, transparent);
}

.composer-inner {
  max-width: 760px;
  margin: 0 auto;
}

/* Status strip */
.status-strip {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--accent);
  margin-bottom: 6px;
  height: 18px;
  padding-left: 2px;
  font-family: ui-monospace, monospace;
  letter-spacing: 0.03em;
  opacity: 0.8;
  transition: opacity 0.2s ease;
}

.status-strip--hidden {
  visibility: hidden;
}

.spin-icon {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* ── Panel transition ────────────────────────────────── */
.panel-enter-active,
.panel-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.panel-enter-from,
.panel-leave-to {
  opacity: 0;
  transform: translateX(16px);
}
</style>
