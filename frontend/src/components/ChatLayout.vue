<script setup lang="ts">
import { ref, watch, nextTick, onMounted, computed } from 'vue'
import { storeToRefs } from 'pinia'
import { Trash2, Loader2, Database, PanelLeft } from 'lucide-vue-next'
import { useChatStore } from '../stores/chatStore'
import ChatMessage from './ChatMessage.vue'
import ChatInput from './ChatInput.vue'
import MemoryPanel from './MemoryPanel.vue'
import ConversationSidebar from './ConversationSidebar.vue'
import aiSlimeAvatar from '../assets/ai_slime_avatar.png'
import { useHeroToChatBubbleFly } from '../composables/useHeroToChatBubbleFly'
import { useChatMessageSplit } from '../composables/useChatMessageSplit'
import type { Message } from '../stores/chatStore'

const chatStore = useChatStore()
const {
  messages,
  status,
  streamingReply,
  isLoading,
  activeVersionId,
  versions,
  pendingLLMError,
  streamingBotIndex,
  transport,
  persistedTurnTick,
} = storeToRefs(chatStore)

const {
  historyTurns,
  activeMessages,
  activeStartIndex,
  activeTurnIndex,
  setActiveTurnByHistorySelection,
  focusNewestTurn,
} = useChatMessageSplit(messages)

const activeAgentEntries = computed(() => {
  const start = activeStartIndex.value
  const active = activeMessages.value
  const out: { msg: Message; globalIdx: number }[] = []
  active.forEach((msg, li) => {
    if (msg.role !== 'user') {
      out.push({ msg, globalIdx: start + li })
    }
  })
  return out
})

const activeUserEntry = computed(() => {
  const start = activeStartIndex.value
  const active = activeMessages.value
  for (let li = active.length - 1; li >= 0; li--) {
    const msg = active[li]!
    if (msg.role === 'user') return { msg, globalIdx: start + li }
  }
  return null as null | { msg: Message; globalIdx: number }
})

const historyTurnEntries = computed(() => {
  return historyTurns.value.map((turn, idx) => {
    const userLocalIndex = turn.messages.findIndex((m) => m.role === 'user')
    const userMsg = userLocalIndex >= 0 ? turn.messages[userLocalIndex]!.text : ''
    return {
      ...turn,
      title: userMsg || 'Untitled',
      userLocalIndex,
      isActive: idx === activeTurnIndex.value,
    }
  })
})

const showEmptyAgentBubble = computed(() => {
  if (activeAgentEntries.value.length > 0) return false
  if (streamingBotIndex.value >= 0) return false
  return activeMessages.value.some((m) => m.role === 'user')
})

const toggleTransport = () => {
  transport.value = transport.value === 'ws' ? 'sse' : 'ws'
}
const logRef = ref<HTMLElement | null>(null)
const historyRailRef = ref<HTMLElement | null>(null)
const heroAvatarRef = ref<HTMLImageElement | null>(null)
const showPanel = ref(false)
const sidebarOpen = ref(localStorage.getItem('agent_sidebar_open') !== '0')
const historyRailOpen = ref(localStorage.getItem('agent_history_rail_open') !== '0')
const heroThinkingText = computed(() => streamingReply.value || status.value || 'AI 思考中')

function toggleSidebar() {
  sidebarOpen.value = !sidebarOpen.value
  localStorage.setItem('agent_sidebar_open', sidebarOpen.value ? '1' : '0')
}

function toggleHistoryRail() {
  historyRailOpen.value = !historyRailOpen.value
  localStorage.setItem('agent_history_rail_open', historyRailOpen.value ? '1' : '0')
}

const scrollHistoryRailToBottom = async () => {
  await nextTick()
  const el = historyRailRef.value
  if (el) {
    el.scrollTop = el.scrollHeight
  }
}

watch(
  () => messages.value,
  () => { scrollHistoryRailToBottom() },
  { deep: true }
)

onMounted(async () => {
  scrollHistoryRailToBottom()
  await Promise.all([chatStore.fetchVersions(), chatStore.fetchChats()])
  // Restore last active chat on first load
  const { activeChatId } = storeToRefs(chatStore)
  if (activeChatId.value) {
    chatStore.switchToChat(activeChatId.value)
  }
})

const handleSend = (payload: { text: string; llmMode: 'auto' | 'gemini' | 'agent' }) => {
  chatStore.sendMessage(payload.text, payload.llmMode)
  focusNewestTurn()
}

const handleTerminate = () => {
  hideBubbleUntilFlyIndex.value = null
  chatStore.terminateMessage()
}

const handleClear = () => {
  if (confirm('Clear all chat history?')) {
    hideBubbleUntilFlyIndex.value = null
    chatStore.clearHistory()
  }
}

const handleFeedback = async (rating: number, messageRef?: string) => {
  const sid = chatStore.sessionId
  if (!sid || !messageRef) return
  const comment = prompt('Optional feedback comment:') ?? ''
  await chatStore.submitFeedback({ sessionId: sid, messageRef, rating, comment })
}

function handleFeedbackForActiveUser(rating: number) {
  const e = activeUserEntry.value
  if (e) handleFeedback(rating, e.msg.messageRef)
}

const activeVersionName = () => {
  const v = versions.value.find((v) => v.version_id === activeVersionId.value)
  return v?.name ?? '—'
}

/** Which message row stays hidden until the fly overlay lands (survives streamingBotIndex clearing on done). */
const hideBubbleUntilFlyIndex = ref<number | null>(null)

watch(streamingBotIndex, (idx, prev) => {
  if (idx >= 0 && prev === -1) hideBubbleUntilFlyIndex.value = idx
})

watch(streamingBotIndex, (idx, prev) => {
  if (idx >= 0 && prev === -1) {
    focusNewestTurn()
  }
})

watch(persistedTurnTick, () => {
  focusNewestTurn()
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
</script>

<template>
  <div class="root-layout">
    <!-- Command Bar -->
    <header class="command-bar">
      <div class="flex items-center gap-3">
        <button
          @click="toggleSidebar"
          class="cmd-btn sidebar-toggle"
          :class="sidebarOpen ? 'cmd-btn--active' : ''"
          title="Toggle sidebar"
        >
          <PanelLeft :size="14" />
        </button>
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

    <!-- Body: sidebar + chat canvas -->
    <div class="body-row">
      <!-- Conversation sidebar -->
      <Transition name="sidebar">
        <ConversationSidebar v-if="sidebarOpen" />
      </Transition>

      <!-- Center stage + history rail -->
      <main ref="logRef" class="chat-canvas">
        <div class="center-stage">
          <div class="center-stage-inner">
            <div class="center-stage-hero-cluster">
            <div class="center-stage-block center-stage-block--agent">
            <div
              v-if="showEmptyAgentBubble"
              class="agent-bubble-placeholder"
              :class="{ 'agent-bubble-placeholder--thinking': isLoading && streamingBotIndex === -1 }"
              role="status"
              aria-live="polite"
            >
              <span
                v-if="isLoading && streamingBotIndex === -1"
                class="agent-placeholder-thinking"
              >{{ heroThinkingText }}</span>
            </div>

            <ChatMessage
              v-for="entry in activeAgentEntries"
              :key="`a-${entry.globalIdx}`"
              class="center-stage-msg"
              hide-role-label
              :role="entry.msg.role"
              :text="entry.msg.text"
              :message-ref="entry.msg.messageRef"
              :feedback-status="entry.msg.feedbackStatus"
              :feedback-rating="entry.msg.feedbackRating"
              :llm-error="entry.msg.llmError"
              :show-actions="pendingLLMError?.messageIndex === entry.globalIdx"
              :streaming="streamingBotIndex === entry.globalIdx"
              :await-fly-reveal="hideBubbleUntilFlyIndex === entry.globalIdx"
              @fix-issue="chatStore.fixIssue()"
              @answer-immediately="chatStore.answerImmediately()"
              @feedback="(rating) => handleFeedback(rating, entry.msg.messageRef)"
            />
            </div>

            <div class="hero-banner hero-banner--stage">
              <div class="hero-activity">
                <span class="hero-activity-dot" :class="isLoading ? 'dot-active' : ''"></span>
              </div>
              <img ref="heroAvatarRef" :src="aiSlimeAvatar" alt="" class="hero-avatar" aria-hidden="true" />
            </div>

            <div v-if="activeUserEntry" class="center-stage-block center-stage-block--user">
            <ChatMessage
              :key="`u-${activeUserEntry.globalIdx}`"
              class="center-stage-msg center-stage-msg--user"
              :role="activeUserEntry.msg.role"
              :text="activeUserEntry.msg.text"
              hide-user-avatar
              :message-ref="activeUserEntry.msg.messageRef"
              :feedback-status="activeUserEntry.msg.feedbackStatus"
              :feedback-rating="activeUserEntry.msg.feedbackRating"
              :llm-error="activeUserEntry.msg.llmError"
              :show-actions="pendingLLMError?.messageIndex === activeUserEntry.globalIdx"
              :streaming="streamingBotIndex === activeUserEntry.globalIdx"
              :await-fly-reveal="hideBubbleUntilFlyIndex === activeUserEntry.globalIdx"
              @fix-issue="chatStore.fixIssue()"
              @answer-immediately="chatStore.answerImmediately()"
              @feedback="handleFeedbackForActiveUser"
            />
            </div>
            </div>
          </div>
        </div>

        <aside
          class="history-rail"
          :class="{ 'history-rail--collapsed': !historyRailOpen }"
          aria-label="Earlier messages"
        >
          <div class="history-rail-header">
            <span class="history-rail-title" :class="{ 'history-rail-title--hidden': !historyRailOpen }">History</span>
            <button
              class="history-rail-toggle"
              :title="historyRailOpen ? 'Collapse history' : 'Expand history'"
              :aria-label="historyRailOpen ? 'Collapse history' : 'Expand history'"
              :aria-expanded="historyRailOpen"
              @click="toggleHistoryRail"
            >
              <span class="history-rail-toggle-icon" :class="{ 'history-rail-toggle-icon--collapsed': !historyRailOpen }">></span>
            </button>
          </div>
          <div v-show="historyRailOpen" ref="historyRailRef" class="history-rail-scroll">
            <div class="history-rail-inner">
              <section
                v-for="turn in historyTurnEntries"
                :key="`turn-${turn.start}`"
                class="history-rail-turn"
                :class="{ 'history-rail-turn--active': turn.isActive }"
              >
                <ChatMessage
                  v-for="(msg, i) in turn.messages"
                  :key="`h-${turn.start}-${i}`"
                  class="history-rail-msg"
                  :role="msg.role"
                  :text="msg.text"
                  :message-ref="msg.messageRef"
                  :feedback-status="msg.feedbackStatus"
                  :feedback-rating="msg.feedbackRating"
                  :llm-error="msg.llmError"
                  :show-actions="pendingLLMError?.messageIndex === (turn.start + i)"
                  :streaming="streamingBotIndex === (turn.start + i)"
                  :await-fly-reveal="hideBubbleUntilFlyIndex === (turn.start + i)"
                  @fix-issue="chatStore.fixIssue()"
                  @answer-immediately="chatStore.answerImmediately()"
                  @feedback="(rating) => handleFeedback(rating, msg.messageRef)"
                />
                <div v-if="turn.userLocalIndex >= 0" class="history-rail-turn-action">
                  <button
                    class="history-rail-switch-btn"
                    type="button"
                    @click="setActiveTurnByHistorySelection(turn.start)"
                  >
                    切換對話
                  </button>
                </div>
              </section>
            </div>
          </div>
        </aside>
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
        <div class="status-strip" :class="(status || isLoading) ? '' : 'status-strip--hidden'">
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

/* ── Chat canvas (center + history rail) ─────────────── */
.chat-canvas {
  flex: 1;
  display: flex;
  flex-direction: row;
  min-height: 0;
  overflow: hidden;
}

.center-stage {
  flex: 1;
  min-width: 0;
  min-height: 0;
  overflow-y: auto;
  padding: 16px 20px 12px;
  scroll-behavior: smooth;
  display: flex;
  flex-direction: column;
}

.center-stage-inner {
  max-width: 760px;
  width: 100%;
  margin-left: auto;
  margin-right: auto;
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  align-items: stretch;
  min-height: 0;
}

.center-stage-hero-cluster {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 14px;
  flex-shrink: 0;
  width: 100%;
  margin-top: auto;
  padding-bottom: 8px;
}

.center-stage-block--agent {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.center-stage-block--user {
  width: 100%;
  display: flex;
  justify-content: center;
}

.agent-bubble-placeholder {
  width: 100%;
  max-width: min(560px, 100%);
  min-height: 72px;
  border-radius: 12px;
  border: 1px dashed rgba(160, 100, 255, 0.38);
  background: rgba(140, 80, 255, 0.06);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 10px 14px;
  box-sizing: border-box;
}

.agent-bubble-placeholder--thinking {
  border-style: solid;
  border-color: rgba(160, 100, 255, 0.3);
  background: rgba(140, 80, 255, 0.08);
  animation: hero-think-pulse 1s ease-in-out infinite;
}

.agent-placeholder-thinking {
  font-size: 12px;
  font-family: ui-monospace, monospace;
  line-height: 1.35;
  color: rgba(206, 180, 255, 0.95);
  text-align: center;
  word-break: break-word;
}

.center-stage-msg {
  width: 100%;
}

.center-stage-msg:not(.center-stage-msg--user) {
  max-width: min(560px, 100%);
}

:deep(.center-stage-msg.msg-row--bot),
:deep(.center-stage-msg.msg-row--err),
:deep(.center-stage-msg.msg-row--thought) {
  justify-content: center;
}

:deep(.center-stage-msg.msg-row--bot .msg-bubble),
:deep(.center-stage-msg.msg-row--err .msg-bubble),
:deep(.center-stage-msg.msg-row--thought .msg-bubble) {
  max-width: 100%;
}

:deep(.center-stage-msg.msg-row--bot .bubble-bot),
:deep(.center-stage-msg.msg-row--err .bubble-err),
:deep(.center-stage-msg.msg-row--thought .bubble-thought) {
  text-align: left;
}

.center-stage-inner .center-stage-msg :deep(.msg-row) {
  margin-bottom: 0;
}

/* center-stage-msg lives on the same node as msg-row--bot (child root), not an ancestor */
:deep(.center-stage-msg.msg-row--bot > .msg-avatar),
:deep(.center-stage-msg.msg-row--err > .msg-avatar),
:deep(.center-stage-msg.msg-row--thought > .msg-avatar) {
  display: none;
}

.center-stage-msg--user {
  width: auto;
  max-width: min(560px, 100%);
}

.center-stage-msg--user :deep(.msg-row--user) {
  width: auto;
  max-width: 100%;
  justify-content: center;
}

.center-stage-msg--user :deep(.msg-bubble) {
  max-width: 100%;
  align-items: center;
}

.center-stage-msg--user :deep(.msg-label) {
  width: 100%;
  text-align: center;
  padding-right: 0;
  box-sizing: border-box;
}

.center-stage-msg--user :deep(.bubble-user) {
  text-align: left;
}

/* ── History rail ────────────────────────────────────── */
.history-rail {
  width: min(300px, 32vw);
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  min-height: 0;
  border-left: 1px solid var(--border);
  background: rgba(10, 11, 15, 0.62);
  transition: width 0.18s ease;
}

.history-rail--collapsed {
  width: 44px;
}

.history-rail--collapsed .history-rail-header {
  justify-content: center;
  padding-left: 6px;
  padding-right: 6px;
}

.history-rail-header {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 10px 12px 8px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-dim);
  font-family: ui-monospace, monospace;
  border-bottom: 1px solid var(--border);
}

.history-rail-title {
  white-space: nowrap;
  transition: opacity 0.15s ease;
}

.history-rail-title--hidden {
  opacity: 0;
  width: 0;
  overflow: hidden;
}

.history-rail-toggle {
  width: 30px;
  height: 30px;
  border-radius: 8px;
  border: 1px solid rgba(0, 229, 255, 0.28);
  background: rgba(0, 229, 255, 0.08);
  color: rgba(180, 244, 255, 0.92);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  flex-shrink: 0;
  padding: 0;
  box-shadow: 0 0 0 1px rgba(0, 229, 255, 0.1) inset;
  transition: color 0.14s ease, border-color 0.14s ease, background 0.14s ease, box-shadow 0.14s ease;
}

.history-rail-toggle:hover {
  color: #d8f9ff;
  border-color: rgba(0, 229, 255, 0.5);
  background: rgba(0, 229, 255, 0.18);
  box-shadow: 0 0 10px rgba(0, 229, 255, 0.22);
}

.history-rail-toggle:focus-visible {
  outline: 2px solid rgba(0, 229, 255, 0.62);
  outline-offset: 2px;
}

.history-rail-toggle-icon {
  display: inline-block;
  font-size: 15px;
  font-weight: 800;
  line-height: 1;
  transform: rotate(180deg);
  transition: transform 0.18s ease;
}

.history-rail-toggle-icon--collapsed {
  transform: rotate(0deg);
}

.history-rail-scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  scroll-behavior: smooth;
  padding: 10px 10px 16px;
}

.history-rail-inner {
  max-width: 100%;
}

.history-rail-turn {
  display: flex;
  flex-direction: column;
  gap: 8px;
  border: 1px solid transparent;
  border-radius: 10px;
  padding: 6px;
  margin-bottom: 8px;
}

.history-rail-turn--active {
  border-color: rgba(0, 229, 255, 0.2);
  background: rgba(0, 229, 255, 0.04);
}

.history-rail-turn-action {
  display: flex;
  justify-content: flex-end;
  padding-right: 4px;
}

.history-rail-switch-btn {
  border: 1px solid rgba(0, 229, 255, 0.22);
  background: rgba(0, 229, 255, 0.06);
  color: var(--accent);
  font-size: 11px;
  border-radius: 7px;
  padding: 4px 8px;
  cursor: pointer;
  transition: background 0.14s ease, border-color 0.14s ease;
}

.history-rail-switch-btn:hover {
  background: rgba(0, 229, 255, 0.12);
  border-color: rgba(0, 229, 255, 0.32);
}

.history-rail-msg :deep(.msg-row) {
  margin-bottom: 14px;
}

.history-rail-msg :deep(.msg-bubble) {
  max-width: 100%;
}

.history-rail-msg :deep(.bubble-bot) {
  font-size: 12.5px;
  padding: 10px 12px;
}

.history-rail-msg :deep(.bubble-user) {
  font-size: 13px;
  padding: 8px 11px;
}

@media (max-width: 900px) {
  .chat-canvas {
    flex-direction: column;
  }

  .history-rail {
    width: 100%;
    max-height: min(240px, 34vh);
    border-left: none;
    border-top: 1px solid var(--border);
    transition: max-height 0.18s ease;
  }

  .history-rail--collapsed {
    width: 100%;
    max-height: 36px;
  }
}

/* ── Hero banner (center stage — fly animation avatar ref) ─ */
.hero-banner {
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 10px 20px 8px;
  gap: 0;
  user-select: none;
  background: transparent;
}

.hero-banner--stage {
  margin: 0;
  padding: 8px 12px 6px;
  align-self: center;
  width: 100%;
  max-width: min(560px, 100%);
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

@keyframes hero-think-pulse {
  0%, 100% { opacity: 0.78; }
  50% { opacity: 1; }
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

/* ── Sidebar toggle button ───────────────────────────── */
.sidebar-toggle {
  padding: 4px 8px;
}

/* ── Sidebar slide transition ────────────────────────── */
.sidebar-enter-active,
.sidebar-leave-active {
  transition: width 0.2s ease, opacity 0.18s ease;
  overflow: hidden;
}

.sidebar-enter-from,
.sidebar-leave-to {
  width: 0 !important;
  opacity: 0;
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
