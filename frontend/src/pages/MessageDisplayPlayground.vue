<script setup lang="ts">
import { ref } from 'vue'
import ChatMessage from '../components/ChatMessage.vue'
import type { LLMErrorPayload } from '../stores/chatStore'

const previewMd = ref(`## Markdown sample

Paragraph with **bold**, \`inline code\`, and a [link](https://example.com).

| Col | Val |
|-----|-----|
| a   | 1   |

\`\`\`ts
const x = 1
\`\`\`

> blockquote

- one
- two
`)

const previewStreaming = ref(false)
const feedbackScenario = ref<'idle' | 'pending' | 'submitted' | 'failed'>('idle')
const previewRating = ref(5)

const llmErr: LLMErrorPayload = {
  is_llm_error: true,
  error_type: 'rate_limit',
  message: 'Upstream limit',
  retry_after_seconds: 30,
}
</script>

<template>
  <div class="play-root">
    <header class="play-bar">
      <span class="play-title">Message display</span>
      <a class="play-link" href="#">← Chat</a>
    </header>

    <div class="play-scroll">
      <div class="play-inner">
        <section class="play-section">
          <h2 class="play-h">Fixtures</h2>
          <ChatMessage role="user" text="Short user line." />
          <ChatMessage
            role="bot"
            :text="`# Heading\n\n- list item\n\n\`\`\`\nconsole.log('hi')\n\`\`\``"
          />
          <ChatMessage role="thought" text="Planning next tool call…" />
          <ChatMessage role="err" text="Something went wrong (no LLM actions)." />
          <ChatMessage
            role="err"
            text="LLM error with recovery actions."
            :llm-error="llmErr"
            :show-actions="true"
          />
        </section>

        <section class="play-section">
          <h2 class="play-h">Feedback states</h2>
          <div class="play-controls">
            <label>
              State
              <select v-model="feedbackScenario" class="play-select">
                <option value="idle">idle</option>
                <option value="pending">pending</option>
                <option value="submitted">submitted</option>
                <option value="failed">failed</option>
              </select>
            </label>
            <label v-if="feedbackScenario === 'submitted'">
              Rating
              <select v-model.number="previewRating" class="play-select">
                <option :value="5">5 helpful</option>
                <option :value="1">1 not helpful</option>
              </select>
            </label>
          </div>
          <ChatMessage
            role="bot"
            text="Message with optional feedback row (needs messageRef)."
            message-ref="playground-ref-1"
            :feedback-status="feedbackScenario"
            :feedback-rating="previewRating"
          />
        </section>

        <section class="play-section">
          <h2 class="play-h">Live markdown / streaming</h2>
          <div class="play-controls">
            <label class="play-check">
              <input v-model="previewStreaming" type="checkbox" />
              streaming (raw text, no markdown)
            </label>
          </div>
          <textarea v-model="previewMd" class="play-textarea" rows="10" spellcheck="false" />
          <ChatMessage role="bot" :text="previewMd" :streaming="previewStreaming" />
        </section>
      </div>
    </div>
  </div>
</template>

<style scoped>
.play-root {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
  background: var(--bg);
}

.play-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 44px;
  flex-shrink: 0;
  background: rgba(17, 19, 24, 0.92);
  backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
}

.play-title {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.12em;
  font-family: ui-monospace, monospace;
  color: var(--text);
}

.play-link {
  font-size: 12px;
  color: var(--text-dim);
  text-decoration: none;
  border-bottom: 1px solid rgba(0, 229, 255, 0.2);
}

.play-link:hover {
  color: var(--accent);
  border-bottom-color: var(--accent);
}

.play-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px 32px;
}

.play-inner {
  max-width: 760px;
  margin: 0 auto;
}

.play-section {
  margin-bottom: 28px;
}

.play-h {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-family: ui-monospace, monospace;
  color: var(--text-dim);
  margin: 0 0 14px;
}

.play-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 14px 20px;
  align-items: center;
  margin-bottom: 12px;
  font-size: 12px;
  color: var(--text-dim);
}

.play-select {
  margin-left: 6px;
  padding: 4px 8px;
  border-radius: 6px;
  border: 1px solid var(--border-bright);
  background: var(--surface-2);
  color: var(--text);
  font-size: 12px;
}

.play-check {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.play-textarea {
  width: 100%;
  margin-bottom: 14px;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid var(--border-bright);
  background: var(--surface);
  color: var(--text);
  font-family: ui-monospace, monospace;
  font-size: 12px;
  line-height: 1.5;
  resize: vertical;
  min-height: 120px;
}

.play-textarea:focus {
  outline: none;
  border-color: rgba(0, 229, 255, 0.35);
}
</style>
