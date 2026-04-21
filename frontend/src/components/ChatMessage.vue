<script setup lang="ts">
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'
import { User, AlertTriangle, Wrench, Zap, Brain } from 'lucide-vue-next'
import aiSlimeAvatar from '../assets/ai_slime_avatar.png'
import type { LLMErrorPayload } from '../stores/chatStore'

const props = defineProps<{
  role: 'user' | 'bot' | 'err' | 'thought'
  text: string
  llmError?: LLMErrorPayload
  messageRef?: string
  feedbackStatus?: 'idle' | 'pending' | 'submitted' | 'failed'
  feedbackRating?: number
  showActions?: boolean
  streaming?: boolean
  /** Hide transcript bubble until fly animation lands (streaming bot only) */
  awaitFlyReveal?: boolean
  /** Omit the small user avatar (center stage layout) */
  hideUserAvatar?: boolean
  /** Omit the YOU / AGENT / … label row */
  hideRoleLabel?: boolean
}>()

const emit = defineEmits<{
  (e: 'fix-issue'): void
  (e: 'answer-immediately'): void
  (e: 'feedback', rating: number): void
}>()

const md = new MarkdownIt({
  linkify: true,
})

const renderedText = computed(() => {
  // Don't run markdown while streaming – the text may be mid-token and
  // produce broken HTML; switch to rendered output only when done.
  if (props.role === 'bot' && !props.streaming) return md.render(props.text)
  return props.text
})

const roleLabel = computed(() => {
  if (props.role === 'user') return 'You'
  if (props.role === 'bot') return 'Agent'
  if (props.role === 'thought') return 'Thinking'
  return 'Error'
})
</script>

<template>
  <div
    class="msg-row"
    :class="{
      'msg-row--user': role === 'user',
      'msg-row--bot': role === 'bot',
      'msg-row--err': role === 'err',
      'msg-row--thought': role === 'thought',
      'msg-row--await-fly': role === 'bot' && awaitFlyReveal,
    }"
  >
    <!-- Avatar (bot / err / thought — left side) -->
    <div
      v-if="role !== 'user'"
      class="msg-avatar"
      :class="{
        'msg-avatar--err': role === 'err',
        'msg-avatar--thought': role === 'thought',
      }"
    >
      <img v-if="role === 'bot'" :src="aiSlimeAvatar" alt="AI" class="avatar-img" />
      <Brain v-else-if="role === 'thought'" :size="13" />
      <AlertTriangle v-else :size="14" />
    </div>

    <!-- Bubble -->
    <div class="msg-bubble">
      <div v-if="!hideRoleLabel" class="msg-label">{{ roleLabel }}</div>

      <div
        v-if="role === 'bot'"
        class="bubble-bot prose-content"
        :data-bot-streaming="streaming ? '' : undefined"
      >
        <span v-if="streaming">{{ text }}</span>
        <span v-else v-html="renderedText"></span>
        <div v-if="!streaming && messageRef" class="feedback-actions">
          <template v-if="feedbackStatus === 'submitted'">
            <span class="feedback-state">Feedback saved ({{ feedbackRating === 5 ? 'Helpful' : 'Not helpful' }})</span>
          </template>
          <template v-else>
            <button class="feedback-btn" :disabled="feedbackStatus === 'pending'" @click="emit('feedback', 5)">Helpful</button>
            <button class="feedback-btn" :disabled="feedbackStatus === 'pending'" @click="emit('feedback', 1)">Not helpful</button>
            <span v-if="feedbackStatus === 'pending'" class="feedback-state">Sending...</span>
            <span v-else-if="feedbackStatus === 'failed'" class="feedback-state feedback-state--error">Send failed, try again.</span>
          </template>
        </div>
      </div>
      <div v-else-if="role === 'err'" class="bubble-err">
        <span>{{ text }}</span>
        <div v-if="llmError?.is_llm_error && showActions" class="err-actions">
          <button class="err-action-btn" @click="emit('fix-issue')">
            <Wrench :size="12" />
            Fix issue
          </button>
          <button class="err-action-btn" @click="emit('answer-immediately')">
            <Zap :size="12" />
            Answer immediately
          </button>
        </div>
      </div>
      <div v-else-if="role === 'thought'" class="bubble-thought">{{ text }}</div>
      <div v-else class="bubble-user">{{ text }}</div>
    </div>

    <!-- Avatar (user right side) -->
    <div v-if="role === 'user' && !hideUserAvatar" class="msg-avatar msg-avatar--user">
      <User :size="14" />
    </div>
  </div>
</template>

<style scoped>
.msg-row {
  display: flex;
  gap: 10px;
  margin-bottom: 22px;
  align-items: flex-start;
}

.msg-row--user {
  flex-direction: row-reverse;
}

.msg-row--await-fly .msg-bubble,
.msg-row--await-fly > .msg-avatar {
  visibility: hidden;
}

/* ── Avatar ──────────────────────────────────────────── */
.msg-avatar {
  width: 30px;
  height: 30px;
  border-radius: 8px;
  flex-shrink: 0;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: 20px;
  background: var(--surface-2);
  border: 1px solid var(--border-bright);
  color: var(--text-dim);
}

.msg-avatar--user {
  background: rgba(var(--accent-rgb), 0.12);
  border-color: rgba(var(--accent-rgb), 0.34);
  color: var(--accent);
}

.msg-avatar--err {
  background: rgba(var(--error-rgb), 0.12);
  border-color: rgba(var(--error-rgb), 0.32);
  color: var(--error);
}

.avatar-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

/* ── Bubble wrapper ──────────────────────────────────── */
.msg-bubble {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-width: 80%;
}

.msg-row--user .msg-bubble {
  align-items: flex-end;
}

/* ── Label ───────────────────────────────────────────── */
.msg-label {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-family: ui-monospace, monospace;
  padding: 0 2px;
}

.msg-row--bot .msg-label,
.msg-row--err .msg-label {
  color: var(--text-dim);
}

.msg-row--user .msg-label {
  color: rgba(var(--accent-rgb), 0.65);
}

/* ── User bubble ─────────────────────────────────────── */
.bubble-user {
  background: rgba(var(--accent-rgb), 0.1);
  border: 1px solid rgba(var(--accent-rgb), 0.28);
  border-radius: 10px 2px 10px 10px;
  padding: 10px 14px;
  font-size: 14px;
  color: var(--text);
  white-space: pre-wrap;
  word-break: normal;
  overflow-wrap: break-word;
  line-height: 1.55;
  max-height: 45vh;
  overflow-y: auto;
}

.bubble-bot::-webkit-scrollbar,
.bubble-user::-webkit-scrollbar,
.bubble-err::-webkit-scrollbar {
  width: 6px;
}

.bubble-bot::-webkit-scrollbar-track,
.bubble-user::-webkit-scrollbar-track,
.bubble-err::-webkit-scrollbar-track {
  background: transparent;
}

.bubble-bot::-webkit-scrollbar-thumb,
.bubble-user::-webkit-scrollbar-thumb,
.bubble-err::-webkit-scrollbar-thumb {
  background: rgba(var(--accent-rgb), 0.26);
  border-radius: 3px;
}

.bubble-bot::-webkit-scrollbar-thumb:hover,
.bubble-user::-webkit-scrollbar-thumb:hover,
.bubble-err::-webkit-scrollbar-thumb:hover {
  background: rgba(var(--accent-rgb), 0.45);
}

/* ── Error bubble ────────────────────────────────────── */
.bubble-err {
  background: rgba(var(--error-rgb), 0.1);
  border: 1px solid rgba(var(--error-rgb), 0.24);
  border-radius: 2px 10px 10px 10px;
  padding: 10px 14px;
  font-size: 13px;
  color: var(--error);
  white-space: pre-wrap;
  word-break: normal;
  overflow-wrap: break-word;
  line-height: 1.55;
  max-height: 45vh;
  overflow-y: auto;
}

/* ── Error action buttons ────────────────────────────── */
.err-actions {
  display: flex;
  gap: 8px;
  margin-top: 10px;
  flex-wrap: wrap;
}

.err-action-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 10px;
  font-size: 11px;
  font-family: ui-monospace, monospace;
  border-radius: 6px;
  border: 1px solid rgba(var(--error-rgb), 0.38);
  color: rgba(255, 255, 255, 0.65);
  background: rgba(var(--error-rgb), 0.08);
  cursor: pointer;
  transition: border-color 0.15s, color 0.15s, background 0.15s;
}

.err-action-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: rgba(var(--accent-rgb), 0.1);
}

/* ── Bot bubble ──────────────────────────────────────── */
.bubble-bot {
  background: var(--surface);
  border: 1px solid var(--border-bright);
  border-left: 2px solid rgba(var(--accent-rgb), 0.38);
  border-radius: 2px 10px 10px 10px;
  padding: 12px 16px;
  font-size: 14px;
  color: var(--text);
  line-height: 1.65;
  word-break: normal;
  overflow-wrap: break-word;
  max-height: 45vh;
  overflow-y: auto;
}

.feedback-actions {
  display: flex;
  gap: 8px;
  margin-top: 10px;
}

.feedback-btn {
  padding: 3px 8px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--text-dim);
  font-size: 11px;
  cursor: pointer;
}

.feedback-btn:hover {
  color: var(--accent);
  border-color: rgba(var(--accent-rgb), 0.34);
}

.feedback-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.feedback-state {
  font-size: 11px;
  color: var(--text-dim);
  padding: 4px 0;
}

.feedback-state--error {
  color: var(--error);
}

/* ── Thought avatar ──────────────────────────────────── */
.msg-avatar--thought {
  background: rgba(var(--secondary-rgb), 0.12);
  border-color: rgba(var(--secondary-rgb), 0.3);
  color: rgba(var(--secondary-rgb), 0.9);
}

/* ── Thought label ───────────────────────────────────── */
.msg-row--thought .msg-label {
  color: rgba(var(--secondary-rgb), 0.7);
}

/* ── Thought bubble ──────────────────────────────────── */
.bubble-thought {
  background: rgba(var(--secondary-rgb), 0.1);
  border: 1px solid rgba(var(--secondary-rgb), 0.24);
  border-left: 2px solid rgba(var(--secondary-rgb), 0.4);
  border-radius: 2px 10px 10px 10px;
  padding: 8px 13px;
  font-size: 12px;
  font-family: ui-monospace, monospace;
  color: rgba(223, 212, 255, 0.78);
  font-style: italic;
  line-height: 1.55;
  word-break: normal;
  overflow-wrap: break-word;
  letter-spacing: 0.01em;
}
</style>

<!-- Global prose styles for v-html rendered markdown -->
<style>
.bubble-bot p { margin: 0.35em 0; }
.bubble-bot p:first-child { margin-top: 0; }
.bubble-bot p:last-child { margin-bottom: 0; }

.bubble-bot pre {
  background: #060709;
  padding: 12px 14px;
  border-radius: 8px;
  overflow-x: auto;
  border: 1px solid rgba(var(--accent-rgb), 0.18);
  margin: 10px 0;
  font-size: 12.5px;
}

.bubble-bot code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.845em;
  background: rgba(var(--accent-rgb), 0.12);
  color: #b8d8ff;
  padding: 0.1em 0.4em;
  border-radius: 4px;
  border: 1px solid rgba(var(--accent-rgb), 0.2);
}

.bubble-bot pre code {
  background: transparent;
  padding: 0;
  border: none;
  color: #c9d1d9;
  font-size: 13px;
}

.bubble-bot ul {
  list-style-type: disc;
  padding-left: 1.4em;
  margin: 6px 0;
}

.bubble-bot ol {
  list-style-type: decimal;
  padding-left: 1.4em;
  margin: 6px 0;
}

.bubble-bot li { margin: 3px 0; }

.bubble-bot h1,
.bubble-bot h2,
.bubble-bot h3 {
  font-weight: 600;
  margin: 0.9em 0 0.3em;
  color: #e2e4ea;
}

.bubble-bot h1 { font-size: 1.2em; }
.bubble-bot h2 { font-size: 1.08em; }
.bubble-bot h3 { font-size: 1em; }

.bubble-bot blockquote {
  border-left: 2px solid rgba(var(--accent-rgb), 0.42);
  padding-left: 12px;
  margin: 8px 0;
  color: #8b90a0;
  font-style: italic;
}

.bubble-bot a {
  color: var(--accent);
  text-decoration: none;
  border-bottom: 1px solid rgba(var(--accent-rgb), 0.4);
}

.bubble-bot a:hover {
  border-bottom-color: var(--accent);
}

.bubble-bot hr {
  border: none;
  border-top: 1px solid rgba(255, 255, 255, 0.07);
  margin: 12px 0;
}

.bubble-bot table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
  margin: 10px 0;
}

.bubble-bot th {
  text-align: left;
  padding: 6px 10px;
  border-bottom: 1px solid rgba(var(--accent-rgb), 0.24);
  color: rgba(var(--accent-rgb), 0.84);
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  font-family: ui-monospace, monospace;
}

.bubble-bot td {
  padding: 6px 10px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}
</style>
