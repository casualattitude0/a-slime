<script setup lang="ts">
import { ref, watch, nextTick, onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { Trash2, Loader2, Database } from 'lucide-vue-next'
import { useChatStore } from '../stores/chatStore'
import ChatMessage from './ChatMessage.vue'
import ChatInput from './ChatInput.vue'
import MemoryPanel from './MemoryPanel.vue'
import aiSlimeAvatar from '../assets/ai_slime_avatar.png'

const chatStore = useChatStore()
const { messages, status, isLoading, activeVersionId, versions, pendingLLMError, streamingBotIndex, transport } = storeToRefs(chatStore)

const toggleTransport = () => {
  transport.value = transport.value === 'ws' ? 'sse' : 'ws'
}
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
          @click="toggleTransport"
          class="cmd-btn transport-toggle"
          :class="transport === 'ws' ? 'cmd-btn--active' : ''"
          :title="transport === 'ws' ? 'Live (WebSocket) — click to switch to SSE' : 'SSE — click to switch to WebSocket'"
          :disabled="isLoading"
        >
          <span class="transport-dot" :class="transport === 'ws' ? 'transport-dot--ws' : 'transport-dot--sse'"></span>
          <span class="cmd-btn-label">{{ transport === 'ws' ? 'Live' : 'SSE' }}</span>
        </button>
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

    <!-- Body: chat canvas -->
    <div class="body-row">
      <!-- Message Canvas -->
      <main class="chat-canvas">
        <!-- Hero banner — always visible -->
        <div class="hero-banner">
          <div class="hero-activity">
            <span class="hero-activity-dot" :class="isLoading ? 'dot-active' : ''"></span>
            <span class="hero-activity-text">{{ status || 'Idle · Waiting for input' }}</span>
          </div>
          <div class="hero-avatar-wrap">
            <img :src="aiSlimeAvatar" alt="Agent" class="hero-avatar" />
            <div class="hero-avatar-ring"></div>
          </div>
          <p class="hero-name">LOCAL AGENT</p>
          <p v-if="messages.length === 0" class="hero-sub">Send a message to begin</p>
        </div>

        <!-- Scrollable messages area -->
        <div ref="logRef" class="messages-scroll">
          <div v-if="messages.length > 0" class="messages-inner">
            <ChatMessage
              v-for="(msg, i) in messages"
              :key="i"
              :role="msg.role"
              :text="msg.text"
              :llm-error="msg.llmError"
              :show-actions="pendingLLMError?.messageIndex === i"
              :streaming="streamingBotIndex === i"
              @fix-issue="chatStore.fixIssue()"
              @answer-immediately="chatStore.answerImmediately()"
            />
          </div>
        </div>
      </main>

    </div>

    <!-- Floating Memory Panel -->
    <Transition name="panel">
      <div v-if="showPanel" class="panel-floating">
        <MemoryPanel />
      </div>
    </Transition>

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
  position: relative;
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

/* ── Transport toggle ─────────────────────────────────── */
.transport-toggle {
  gap: 6px;
}

.transport-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
  transition: background 0.2s ease, box-shadow 0.2s ease;
}

.transport-dot--ws {
  background: var(--accent);
  box-shadow: 0 0 6px var(--accent-glow);
  animation: dot-pulse 1.4s ease-in-out infinite;
}

.transport-dot--sse {
  background: rgba(255, 200, 100, 0.7);
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
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

/* ── Messages scroll ─────────────────────────────────── */
.messages-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px 12px;
  scroll-behavior: smooth;
}

.messages-inner {
  max-width: 760px;
  margin: 0 auto;
}

/* ── Hero banner (always visible) ───────────────────── */
.hero-banner {
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 28px 20px 20px;
  gap: 0;
  user-select: none;
  border-bottom: 1px solid var(--border);
  background: rgba(17, 19, 24, 0.5);
}

.hero-activity {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 14px;
  font-size: 11px;
  font-family: ui-monospace, monospace;
  letter-spacing: 0.07em;
  color: var(--accent);
  opacity: 0.75;
}

.hero-activity-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: rgba(0, 229, 255, 0.35);
  flex-shrink: 0;
  transition: background 0.3s ease, box-shadow 0.3s ease;
}

.hero-activity-dot.dot-active {
  background: var(--accent);
  box-shadow: 0 0 8px var(--accent-glow);
  animation: dot-pulse 1.4s ease-in-out infinite;
}

.hero-avatar-wrap {
  position: relative;
  width: 80px;
  height: 80px;
  margin-bottom: 14px;
}

.hero-avatar {
  width: 80px;
  height: 80px;
  border-radius: 22px;
  object-fit: cover;
  border: 1px solid rgba(0, 229, 255, 0.22);
  box-shadow: 0 0 24px rgba(0, 229, 255, 0.1), 0 4px 20px rgba(0, 0, 0, 0.45);
  position: relative;
  z-index: 1;
}

.hero-avatar-ring {
  position: absolute;
  inset: -6px;
  border-radius: 28px;
  border: 1px solid rgba(0, 229, 255, 0.1);
  pointer-events: none;
  z-index: 0;
}

.hero-name {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.14em;
  color: var(--text);
  font-family: ui-monospace, monospace;
  margin: 0 0 4px;
}

.hero-sub {
  font-size: 11px;
  color: var(--text-dim);
  margin: 0;
  letter-spacing: 0.02em;
}

/* ── Floating memory panel ───────────────────────────── */
.panel-floating {
  position: absolute;
  top: 56px;
  right: 16px;
  width: min(360px, calc(100vw - 32px));
  height: min(70vh, 640px);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  background: rgba(12, 14, 18, 0.95);
  backdrop-filter: blur(18px);
  box-shadow: 0 18px 52px rgba(0, 0, 0, 0.42);
  z-index: 35;
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
