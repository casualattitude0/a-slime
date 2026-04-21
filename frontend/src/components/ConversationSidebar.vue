<script setup lang="ts">
import { ref, computed, nextTick } from 'vue'
import { storeToRefs } from 'pinia'
import { Plus, Trash2, Pencil, Check, X, MessageSquare } from 'lucide-vue-next'
import { useChatStore } from '../stores/chatStore'
import type { ChatEntry } from '../stores/chatStore'

const chatStore = useChatStore()
const { chats, activeChatId, isLoading } = storeToRefs(chatStore)

// ── Rename state ─────────────────────────────────────────────────────────────
const renamingId = ref<string | null>(null)
const renameValue = ref('')
const renameInputRef = ref<HTMLInputElement | null>(null)

function startRename(chat: ChatEntry) {
  renamingId.value = chat.chat_id
  renameValue.value = chat.title
  nextTick(() => renameInputRef.value?.focus())
}

async function commitRename(chatId: string) {
  if (renameValue.value.trim()) {
    await chatStore.renameChat(chatId, renameValue.value.trim())
  }
  renamingId.value = null
}

function cancelRename() {
  renamingId.value = null
}

// ── Delete confirm ────────────────────────────────────────────────────────────
const confirmDeleteId = ref<string | null>(null)

async function handleDelete(chatId: string) {
  if (confirmDeleteId.value === chatId) {
    await chatStore.deleteChat(chatId)
    confirmDeleteId.value = null
  } else {
    confirmDeleteId.value = chatId
    // Auto-cancel after 3s
    setTimeout(() => {
      if (confirmDeleteId.value === chatId) confirmDeleteId.value = null
    }, 3000)
  }
}

// ── Chat grouping ─────────────────────────────────────────────────────────────
interface ChatGroup {
  label: string
  items: ChatEntry[]
}

function parseDateMs(value?: string): number | null {
  if (!value) return null
  const primary = Date.parse(value)
  if (Number.isFinite(primary)) return primary
  const normalized = value.replace(' ', 'T')
  const fallback = Date.parse(normalized)
  return Number.isFinite(fallback) ? fallback : null
}

function getLastEditedMs(chat: ChatEntry): number {
  return parseDateMs(chat.updated_at) ?? parseDateMs(chat.created_at) ?? 0
}

const groupedChats = computed<ChatGroup[]>(() => {
  const sortedChats = [...chats.value].sort((a, b) => getLastEditedMs(b) - getLastEditedMs(a))
  const day = 86_400_000
  const now = new Date()
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const startOfYesterday = startOfToday - day
  const startOfSevenDays = startOfToday - 6 * day

  const today: ChatEntry[] = []
  const yesterday: ChatEntry[] = []
  const week: ChatEntry[] = []
  const older: ChatEntry[] = []

  for (const c of sortedChats) {
    const editedAt = getLastEditedMs(c)
    if (editedAt >= startOfToday) today.push(c)
    else if (editedAt >= startOfYesterday) yesterday.push(c)
    else if (editedAt >= startOfSevenDays) week.push(c)
    else older.push(c)
  }

  const groups: ChatGroup[] = []
  if (today.length) groups.push({ label: 'Today', items: today })
  if (yesterday.length) groups.push({ label: 'Yesterday', items: yesterday })
  if (week.length) groups.push({ label: 'Last 7 Days', items: week })
  if (older.length) groups.push({ label: 'Older', items: older })
  return groups
})

async function handleSwitch(chatId: string) {
  if (activeChatId.value === chatId || isLoading.value) return
  await chatStore.switchToChat(chatId)
}

async function handleNewChat() {
  await chatStore.createNewChat()
}
</script>

<template>
  <div class="sidebar">
    <!-- Header -->
    <div class="sidebar-header">
      <span class="sidebar-title">Chats</span>
      <button class="new-btn" @click="handleNewChat" :disabled="isLoading" title="New chat">
        <Plus :size="15" />
      </button>
    </div>

    <!-- Chat list -->
    <div class="sidebar-list">
      <div v-if="chats.length === 0" class="empty-state">
        <MessageSquare :size="20" class="empty-icon" />
        <span>No conversations yet</span>
      </div>

      <template v-for="group in groupedChats" :key="group.label">
        <div class="group-label">{{ group.label }}</div>

        <div
          v-for="chat in group.items"
          :key="chat.chat_id"
          class="chat-item"
          :class="{ 'chat-item--active': activeChatId === chat.chat_id }"
          @click="handleSwitch(chat.chat_id)"
        >
          <!-- Rename input -->
          <template v-if="renamingId === chat.chat_id">
            <input
              ref="renameInputRef"
              v-model="renameValue"
              class="rename-input"
              @keydown.enter="commitRename(chat.chat_id)"
              @keydown.esc="cancelRename"
              @blur="commitRename(chat.chat_id)"
              @click.stop
            />
            <button class="icon-btn icon-btn--ok" @click.stop="commitRename(chat.chat_id)" title="Save">
              <Check :size="12" />
            </button>
            <button class="icon-btn icon-btn--cancel" @click.stop="cancelRename" title="Cancel">
              <X :size="12" />
            </button>
          </template>

          <!-- Normal row -->
          <template v-else>
            <span class="chat-title" :title="chat.title">{{ chat.title }}</span>
            <div class="chat-actions">
              <button
                class="icon-btn"
                @click.stop="startRename(chat)"
                title="Rename"
              >
                <Pencil :size="12" />
              </button>
              <button
                class="icon-btn"
                :class="confirmDeleteId === chat.chat_id ? 'icon-btn--danger-active' : 'icon-btn--danger'"
                @click.stop="handleDelete(chat.chat_id)"
                :title="confirmDeleteId === chat.chat_id ? 'Click again to confirm' : 'Delete'"
              >
                <Trash2 :size="12" />
              </button>
            </div>
          </template>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.sidebar {
  width: 240px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  height: 100%;
  background: rgba(10, 11, 15, 0.96);
  border-right: 1px solid var(--border);
  overflow: hidden;
}

/* ── Header ──────────────────────────────────────────── */
.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px 10px;
  flex-shrink: 0;
  border-bottom: 1px solid var(--border);
}

.sidebar-title {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-dim);
  font-family: ui-monospace, monospace;
}

.new-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border-radius: 6px;
  border: 1px solid var(--border-bright);
  background: transparent;
  color: var(--text-dim);
  cursor: pointer;
  transition: all 0.14s ease;
  flex-shrink: 0;
}

.new-btn:hover {
  color: var(--accent);
  border-color: rgba(var(--accent-rgb), 0.42);
  background: var(--accent-soft);
}

.new-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

/* ── List ────────────────────────────────────────────── */
.sidebar-list {
  flex: 1;
  overflow-y: auto;
  padding: 6px 8px 16px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 32px 12px;
  color: var(--text-dim);
  font-size: 11px;
  text-align: center;
  font-family: ui-monospace, monospace;
}

.empty-icon {
  opacity: 0.35;
}

/* ── Group label ─────────────────────────────────────── */
.group-label {
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-dim);
  font-family: ui-monospace, monospace;
  padding: 10px 6px 4px;
  opacity: 0.5;
}

/* ── Chat item ───────────────────────────────────────── */
.chat-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 7px 8px;
  border-radius: 7px;
  cursor: pointer;
  transition: background 0.12s ease;
  min-height: 34px;
  position: relative;
  border: 1px solid transparent;
}

.chat-item:hover {
  background: var(--surface-2);
}

.chat-item--active {
  background: var(--surface-2);
  border-color: rgba(var(--accent-rgb), 0.24);
}

.chat-item--active .chat-title {
  color: var(--text);
}

/* Show actions only on hover */
.chat-item .chat-actions {
  display: none;
  gap: 2px;
  flex-shrink: 0;
}

.chat-item:hover .chat-actions,
.chat-item--active .chat-actions {
  display: flex;
}

.chat-title {
  flex: 1;
  font-size: 12.5px;
  color: rgba(226, 228, 234, 0.7);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
  min-width: 0;
}

/* ── Icon buttons ────────────────────────────────────── */
.icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 5px;
  border: none;
  background: transparent;
  color: var(--text-dim);
  cursor: pointer;
  transition: all 0.12s ease;
  flex-shrink: 0;
}

.icon-btn:hover {
  background: var(--surface-3);
  color: var(--text);
}

.icon-btn--ok:hover {
  color: var(--success);
  background: rgba(0, 255, 163, 0.08);
}

.icon-btn--cancel:hover {
  color: var(--text-dim);
}

.icon-btn--danger:hover {
  color: var(--error);
  background: var(--error-soft);
}

.icon-btn--danger-active {
  color: var(--error);
  background: var(--error-soft);
  animation: danger-pulse 0.8s ease-in-out infinite;
}

@keyframes danger-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}

/* ── Rename input ────────────────────────────────────── */
.rename-input {
  flex: 1;
  min-width: 0;
  background: var(--surface-3);
  border: 1px solid rgba(var(--accent-rgb), 0.42);
  border-radius: 5px;
  color: var(--text);
  font-size: 12px;
  padding: 3px 6px;
  outline: none;
  font-family: inherit;
}

.rename-input:focus {
  border-color: rgba(var(--accent-rgb), 0.62);
}
</style>
