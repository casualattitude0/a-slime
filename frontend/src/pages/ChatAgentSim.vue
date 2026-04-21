<script setup lang="ts">
import { ref, watch, nextTick, computed, onUnmounted } from 'vue'
import { Trash2, Loader2 } from 'lucide-vue-next'
import ChatMessage from '../components/ChatMessage.vue'
import ChatInput from '../components/ChatInput.vue'
import type { Message } from '../stores/chatStore'
import aiSlimeAvatar from '../assets/ai_slime_avatar.png'
import { useHeroToChatBubbleFly } from '../composables/useHeroToChatBubbleFly'

const messages = ref<Message[]>([])
const isLoading = ref(false)
const status = ref('')
const streamingReply = ref('')
const streamingBotIndex = ref(-1)
const logRef = ref<HTMLElement | null>(null)
const heroAvatarRef = ref<HTMLImageElement | null>(null)

let streamTimer: ReturnType<typeof setInterval> | null = null

const hideBubbleUntilFlyIndex = ref<number | null>(null)

watch(streamingBotIndex, (idx, prev) => {
  if (idx >= 0 && prev === -1) hideBubbleUntilFlyIndex.value = idx
})

useHeroToChatBubbleFly({
  streamingBotIndex,
  streamingReply,
  logRef,
  avatarRef: heroAvatarRef,
  onFlyArrived: () => {
    hideBubbleUntilFlyIndex.value = null
  },
})

const heroThinkingText = computed(
  () => streamingReply.value || status.value || 'Simulation',
)

const scrollToBottom = async () => {
  await nextTick()
  if (logRef.value) logRef.value.scrollTop = logRef.value.scrollHeight
}

watch(
  () => messages.value,
  () => {
    scrollToBottom()
  },
  { deep: true },
)

watch(streamingBotIndex, () => {
  scrollToBottom()
})

function stopStream() {
  if (streamTimer !== null) {
    clearInterval(streamTimer)
    streamTimer = null
  }
}

onUnmounted(() => {
  stopStream()
})

function handleSend(payload: { text: string; llmMode: 'auto' | 'gemini' | 'agent' }) {
  const { text } = payload
  if (!text.trim() || isLoading.value) return

  messages.value.push({ role: 'user', text })
  isLoading.value = true
  status.value = 'Simulated agent…'
  streamingReply.value = ''

  const full = text
  const botIndex = messages.value.length
  messages.value.push({ role: 'bot', text: '' })
  streamingBotIndex.value = botIndex

  let i = 0
  const finish = () => {
    stopStream()
    streamingBotIndex.value = -1
    streamingReply.value = ''
    status.value = ''
    isLoading.value = false
  }

  const tick = () => {
    const step = Math.max(1, Math.ceil(full.length / 36))
    i = Math.min(full.length, i + step)
    const slice = full.slice(0, i)
    messages.value[botIndex]!.text = slice
    streamingReply.value = slice

    if (i >= full.length) finish()
  }

  // Defer first tick so Vue paints [data-bot-streaming] before tick() can call finish()
  // on short messages (otherwise streamingBotIndex flips -1→n→-1 in one sync turn and fly skips).
  requestAnimationFrame(() => {
    tick()
    if (i < full.length) streamTimer = setInterval(tick, 28)
  })
}

function handleTerminate() {
  stopStream()
  hideBubbleUntilFlyIndex.value = null
  streamingBotIndex.value = -1
  streamingReply.value = ''
  status.value = ''
  isLoading.value = false
}

function handleClear() {
  if (!messages.value.length) return
  if (confirm('Clear simulation chat?')) {
    handleTerminate()
    messages.value = []
  }
}
</script>

<template>
  <div class="root-layout">
    <header class="command-bar">
      <div class="flex items-center gap-3">
        <div class="flex items-center gap-2">
          <span class="status-dot" :class="isLoading ? 'dot-active' : ''"></span>
          <span class="agent-name">AGENT SIM</span>
        </div>
        <span class="version-badge">local echo</span>
      </div>
      <div class="flex items-center gap-1">
        <a class="cmd-btn cmd-link" href="#">Chat</a>
        <button
          type="button"
          class="cmd-btn cmd-btn--danger"
          :disabled="isLoading"
          title="Clear"
          @click="handleClear"
        >
          <Trash2 :size="14" />
        </button>
      </div>
    </header>

    <main class="chat-canvas">
      <div ref="logRef" class="messages-scroll">
        <div class="messages-inner">
          <ChatMessage
            v-for="(msg, i) in messages"
            :key="i"
            :role="msg.role"
            :text="msg.text"
            :streaming="streamingBotIndex === i"
            :await-fly-reveal="hideBubbleUntilFlyIndex === i"
          />
        </div>
      </div>
    </main>

    <footer class="composer-footer">
      <div class="composer-inner">
        <div class="hero-banner">
          <div v-if="isLoading && streamingBotIndex === -1" class="hero-thinking-bubble">{{ heroThinkingText }}</div>
          <div class="hero-activity">
            <span class="hero-activity-dot" :class="isLoading ? 'dot-active' : ''"></span>
          </div>
          <img ref="heroAvatarRef" :src="aiSlimeAvatar" alt="Agent" class="hero-avatar" />
        </div>

        <div class="status-strip" :class="status || isLoading ? '' : 'status-strip--hidden'">
          <Loader2 :size="11" class="spin-icon" />
          <span>{{ status || 'AI 思考中' }}</span>
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

.cmd-btn:disabled {
  opacity: 0.38;
  cursor: not-allowed;
}

.cmd-btn--danger:hover {
  color: var(--error);
  background: var(--error-soft);
  border-color: rgba(255, 77, 106, 0.22);
}

.cmd-link {
  text-decoration: none;
  font-family: ui-monospace, monospace;
  font-size: 11px;
  letter-spacing: 0.06em;
}

.chat-canvas {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

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

.hero-banner {
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 10px 20px 8px;
  gap: 0;
  user-select: none;
  border-top: 1px solid var(--border);
  margin-bottom: 8px;
  background: transparent;
}

.hero-activity {
  display: flex;
  align-items: center;
  gap: 0;
  margin-bottom: 8px;
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

.hero-avatar {
  display: block;
  width: auto;
  height: auto;
  max-width: 80px;
  max-height: 80px;
  object-fit: contain;
  border: none;
  box-shadow: none;
  background: none;
  margin-bottom: 4px;
}

.hero-thinking-bubble {
  margin-bottom: 8px;
  padding: 6px 12px;
  border-radius: 12px;
  border: 1px solid rgba(160, 100, 255, 0.3);
  background: rgba(140, 80, 255, 0.08);
  color: rgba(206, 180, 255, 0.95);
  font-size: 12px;
  font-family: ui-monospace, monospace;
  line-height: 1.2;
  animation: hero-think-pulse 1s ease-in-out infinite;
}

@keyframes hero-think-pulse {
  0%, 100% { opacity: 0.78; }
  50% { opacity: 1; }
}

.composer-footer {
  flex-shrink: 0;
  padding: 8px 20px 20px;
  background: linear-gradient(to top, var(--bg) 55%, transparent);
}

.composer-inner {
  max-width: 760px;
  margin: 0 auto;
}

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
</style>

<style>
.bubble-fly-clone {
  position: fixed;
  z-index: 9999;
  max-width: min(560px, calc(100vw - 24px));
  max-height: min(72vh, 520px);
  overflow-x: hidden;
  overflow-y: auto;
  padding: 10px 14px;
  border-radius: 12px;
  border: 1px solid rgba(160, 100, 255, 0.38);
  background: rgba(140, 80, 255, 0.14);
  color: rgba(206, 180, 255, 0.96);
  font-size: 12px;
  font-family: ui-monospace, monospace;
  line-height: 1.35;
  white-space: pre-wrap;
  word-break: break-word;
  box-shadow:
    0 4px 18px rgba(0, 0, 0, 0.35),
    0 0 0 1px rgba(0, 229, 255, 0.06) inset;
  pointer-events: none;
  opacity: 0.9;
  will-change: transform;
}
</style>
