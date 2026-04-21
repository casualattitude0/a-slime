<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import {
  Layers,
  Brain,
  FileText,
  ChevronRight,
  Check,
  Trash2,
  Plus,
  Loader2,
  AlertTriangle,
  X,
  RefreshCw,
} from 'lucide-vue-next'
import { useChatStore } from '../stores/chatStore'
import { formatModelProfileLabel } from '../utils/modelProfile'

const chatStore = useChatStore()
const {
  versions,
  availableProfiles,
  memoryItems,
  memoryLoading,
  ragItems,
  ragLoading,
  isLoading,
} = storeToRefs(chatStore)

type Tab = 'versions' | 'memory' | 'rag'
const activeTab = ref<Tab>('versions')

const newVersionName = ref('')
const newVersionProfile = ref('default')
const showNewVersionForm = ref(false)
const actionLoading = ref<string | null>(null)
const confirmDeleteAll = ref<'' | 'memory' | 'rag' | 'all'>('')

function fmtDate(iso: string) {
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

async function handleSwitchVersion(versionId: string) {
  if (actionLoading.value) return
  actionLoading.value = `switch-${versionId}`
  await chatStore.switchVersion(versionId)
  actionLoading.value = null
}

async function handleDeleteVersion(versionId: string) {
  if (actionLoading.value) return
  actionLoading.value = `del-ver-${versionId}`
  await chatStore.deleteVersion(versionId)
  actionLoading.value = null
}

async function handleCreateVersion() {
  const name = newVersionName.value.trim()
  if (!name || actionLoading.value) return
  actionLoading.value = 'create'
  await chatStore.createVersion(name, newVersionProfile.value)
  newVersionName.value = ''
  showNewVersionForm.value = false
  actionLoading.value = null
}

async function loadTab(tab: Tab) {
  activeTab.value = tab
  if (tab === 'memory') await chatStore.fetchMemoryItems()
  if (tab === 'rag') await chatStore.fetchRagItems()
}

async function handleDeleteMemory(id: string) {
  if (actionLoading.value) return
  actionLoading.value = `del-mem-${id}`
  await chatStore.deleteMemoryItem(id)
  actionLoading.value = null
}

async function handleDeleteAllMemory() {
  if (actionLoading.value) return
  actionLoading.value = 'del-mem-all'
  await chatStore.deleteAllMemory()
  confirmDeleteAll.value = ''
  actionLoading.value = null
}

async function handleDeleteRag(id: string) {
  if (actionLoading.value) return
  actionLoading.value = `del-rag-${id}`
  await chatStore.deleteRagItem(id)
  actionLoading.value = null
}

async function handleDeleteAllRag() {
  if (actionLoading.value) return
  actionLoading.value = 'del-rag-all'
  await chatStore.deleteAllRag()
  confirmDeleteAll.value = ''
  actionLoading.value = null
}

async function handleDeleteAll() {
  if (actionLoading.value) return
  actionLoading.value = 'del-all'
  await chatStore.deleteAllData()
  confirmDeleteAll.value = ''
  actionLoading.value = null
}

onMounted(async () => {
  await chatStore.fetchVersions()
})
</script>

<template>
  <div class="panel-root">
    <!-- Tab bar -->
    <div class="tab-bar">
      <button
        v-for="tab in (['versions', 'memory', 'rag'] as const)"
        :key="tab"
        @click="loadTab(tab)"
        class="tab-btn"
        :class="activeTab === tab ? 'tab-btn--active' : ''"
      >
        <Layers v-if="tab === 'versions'" :size="12" />
        <Brain v-else-if="tab === 'memory'" :size="12" />
        <FileText v-else :size="12" />
        {{ tab === 'versions' ? '版本' : tab === 'memory' ? '記憶' : 'RAG' }}
      </button>
    </div>

    <!-- ── Versions tab ─────────────────────────────────── -->
    <div v-if="activeTab === 'versions'" class="tab-content">
      <div
        v-for="v in versions"
        :key="v.version_id"
        class="ver-card"
        :class="v.is_active ? 'ver-card--active' : ''"
      >
        <div class="ver-card-body">
          <div class="ver-info">
            <div class="ver-name-row">
              <Check v-if="v.is_active" :size="11" class="check-icon" />
              <span class="ver-name">{{ v.name }}</span>
            </div>
            <div class="ver-meta">{{ formatModelProfileLabel(v.model_profile) }}</div>
            <div class="ver-date">{{ fmtDate(v.created_at) }}</div>
          </div>
          <div class="ver-actions">
            <button
              v-if="!v.is_active"
              @click="handleSwitchVersion(v.version_id)"
              :disabled="!!actionLoading"
              class="icon-btn icon-btn--accent"
              title="切換至此版本"
            >
              <Loader2 v-if="actionLoading === `switch-${v.version_id}`" :size="12" class="spin" />
              <ChevronRight v-else :size="12" />
            </button>
            <button
              @click="handleDeleteVersion(v.version_id)"
              :disabled="!!actionLoading || (v.is_active && versions.length === 1)"
              class="icon-btn icon-btn--danger"
              title="刪除版本"
            >
              <Loader2 v-if="actionLoading === `del-ver-${v.version_id}`" :size="12" class="spin" />
              <Trash2 v-else :size="12" />
            </button>
          </div>
        </div>
      </div>

      <!-- New version form -->
      <div v-if="showNewVersionForm" class="new-ver-form">
        <input
          v-model="newVersionName"
          placeholder="Version name"
          @keyup.enter="handleCreateVersion"
          class="panel-input"
        />
        <select v-model="newVersionProfile" class="panel-input">
          <option v-for="p in availableProfiles" :key="p" :value="p">
            {{ formatModelProfileLabel(p) }}
          </option>
        </select>
        <div class="flex gap-1.5">
          <button
            @click="handleCreateVersion"
            :disabled="!newVersionName.trim() || !!actionLoading"
            class="form-btn form-btn--primary"
          >
            <Loader2 v-if="actionLoading === 'create'" :size="11" class="spin" />
            <span>建立</span>
          </button>
          <button @click="showNewVersionForm = false" class="form-btn form-btn--ghost">
            <X :size="11" />
          </button>
        </div>
      </div>

      <button
        v-if="!showNewVersionForm"
        @click="showNewVersionForm = true"
        class="add-btn"
      >
        <Plus :size="11" />
        新增版本
      </button>
    </div>

    <!-- ── Memory tab ──────────────────────────────────── -->
    <div v-else-if="activeTab === 'memory'" class="tab-content-flex">
      <div class="list-toolbar">
        <span class="list-count">{{ memoryItems.length }} 筆記憶</span>
        <div class="flex gap-1">
          <button @click="chatStore.fetchMemoryItems()" :disabled="memoryLoading" class="icon-btn">
            <RefreshCw :size="11" :class="memoryLoading ? 'spin' : ''" />
          </button>
          <button
            v-if="memoryItems.length > 0"
            @click="confirmDeleteAll = 'memory'"
            :disabled="!!actionLoading"
            class="text-btn text-btn--danger"
          >清空</button>
        </div>
      </div>

      <div v-if="memoryLoading" class="list-loading">
        <Loader2 :size="18" class="spin accent-icon" />
      </div>
      <div v-else-if="memoryItems.length === 0" class="list-empty">無記憶項目</div>
      <div v-else class="list-scroll">
        <div v-for="item in memoryItems" :key="item.id" class="list-item">
          <div class="list-item-body">
            <p class="list-item-text">{{ item.content }}</p>
            <div class="list-item-meta">
              <span v-if="item.metadata?.ts">{{ fmtDate(item.metadata.ts) }}</span>
              <span v-if="item.metadata?.tags" class="meta-tag">{{ item.metadata.tags }}</span>
            </div>
          </div>
          <button
            @click="handleDeleteMemory(item.id)"
            :disabled="!!actionLoading"
            class="icon-btn icon-btn--danger"
          >
            <Loader2 v-if="actionLoading === `del-mem-${item.id}`" :size="11" class="spin" />
            <Trash2 v-else :size="11" />
          </button>
        </div>
      </div>
    </div>

    <!-- ── RAG tab ─────────────────────────────────────── -->
    <div v-else-if="activeTab === 'rag'" class="tab-content-flex">
      <div class="list-toolbar">
        <span class="list-count">{{ ragItems.length }} 筆文件</span>
        <div class="flex gap-1">
          <button @click="chatStore.fetchRagItems()" :disabled="ragLoading" class="icon-btn">
            <RefreshCw :size="11" :class="ragLoading ? 'spin' : ''" />
          </button>
          <button
            v-if="ragItems.length > 0"
            @click="confirmDeleteAll = 'rag'"
            :disabled="!!actionLoading"
            class="text-btn text-btn--danger"
          >清空</button>
        </div>
      </div>

      <div v-if="ragLoading" class="list-loading">
        <Loader2 :size="18" class="spin accent-icon" />
      </div>
      <div v-else-if="ragItems.length === 0" class="list-empty">無 RAG 文件</div>
      <div v-else class="list-scroll">
        <div v-for="item in ragItems" :key="item.id" class="list-item">
          <div class="list-item-body">
            <p class="list-item-text">{{ item.content }}</p>
            <p v-if="item.metadata?.source" class="list-item-source">{{ item.metadata.source }}</p>
          </div>
          <button
            @click="handleDeleteRag(item.id)"
            :disabled="!!actionLoading"
            class="icon-btn icon-btn--danger"
          >
            <Loader2 v-if="actionLoading === `del-rag-${item.id}`" :size="11" class="spin" />
            <Trash2 v-else :size="11" />
          </button>
        </div>
      </div>
    </div>

    <!-- ── Danger zone ─────────────────────────────────── -->
    <div class="danger-zone">
      <button
        @click="confirmDeleteAll = 'all'"
        :disabled="!!actionLoading || isLoading"
        class="danger-btn"
      >
        <AlertTriangle :size="11" />
        全部清除
      </button>
    </div>

    <!-- ── Confirm overlay ─────────────────────────────── -->
    <Teleport to="body">
      <div
        v-if="confirmDeleteAll"
        class="overlay"
        @click.self="confirmDeleteAll = ''"
      >
        <div class="confirm-dialog">
          <div class="confirm-header">
            <AlertTriangle :size="16" class="error-icon" />
            <p class="confirm-msg">
              {{ confirmDeleteAll === 'all'
                ? '確認全部清除？此操作不可復原。'
                : confirmDeleteAll === 'memory'
                  ? '確認清空所有記憶？'
                  : '確認清空所有 RAG 文件？' }}
            </p>
          </div>
          <div class="confirm-actions">
            <button @click="confirmDeleteAll = ''" class="form-btn form-btn--ghost">取消</button>
            <button
              @click="
                confirmDeleteAll === 'all'
                  ? handleDeleteAll()
                  : confirmDeleteAll === 'memory'
                    ? handleDeleteAllMemory()
                    : handleDeleteAllRag()
              "
              :disabled="!!actionLoading"
              class="form-btn form-btn--error"
            >
              <Loader2
                v-if="actionLoading === 'del-all' || actionLoading === 'del-mem-all' || actionLoading === 'del-rag-all'"
                :size="11"
                class="spin"
              />
              確認刪除
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.panel-root {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--surface);
  overflow: hidden;
}

/* ── Tab bar ─────────────────────────────────────────── */
.tab-bar {
  display: flex;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.tab-btn {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  padding: 9px 4px;
  font-size: 11px;
  font-weight: 500;
  color: var(--text-dim);
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  cursor: pointer;
  transition: all 0.14s ease;
  font-family: ui-monospace, monospace;
  letter-spacing: 0.03em;
}

.tab-btn:hover {
  color: var(--text);
  background: rgba(255, 255, 255, 0.03);
}

.tab-btn--active {
  color: var(--accent);
  border-bottom-color: var(--accent);
  background: rgba(var(--accent-rgb), 0.12);
}

/* ── Tab content ─────────────────────────────────────── */
.tab-content {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.tab-content-flex {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

/* ── Version card ────────────────────────────────────── */
.ver-card {
  border-radius: 8px;
  border: 1px solid var(--border-bright);
  padding: 9px 10px;
  background: var(--surface-2);
  transition: border-color 0.14s ease;
}

.ver-card:hover {
  border-color: rgba(255, 255, 255, 0.16);
}

.ver-card--active {
  border-color: rgba(var(--accent-rgb), 0.38);
  background: rgba(var(--accent-rgb), 0.12);
}

.ver-card-body {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
}

.ver-info {
  flex: 1;
  min-width: 0;
}

.ver-name-row {
  display: flex;
  align-items: center;
  gap: 5px;
}

.check-icon {
  color: var(--accent);
  flex-shrink: 0;
}

.ver-name {
  font-size: 12px;
  font-weight: 600;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.ver-meta {
  font-size: 10.5px;
  color: var(--text-dim);
  margin-top: 2px;
  font-family: ui-monospace, monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.ver-date {
  font-size: 10px;
  color: rgba(107, 114, 128, 0.6);
  margin-top: 1px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.ver-actions {
  display: flex;
  align-items: center;
  gap: 3px;
  flex-shrink: 0;
}

/* ── New version form ────────────────────────────────── */
.new-ver-form {
  display: flex;
  flex-direction: column;
  gap: 6px;
  border-radius: 8px;
  border: 1px solid var(--border-bright);
  padding: 10px;
  background: var(--surface-2);
}

.panel-input {
  width: 100%;
  background: var(--surface-3);
  border: 1px solid var(--border-bright);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 12px;
  color: var(--text);
  font-family: inherit;
  outline: none;
  transition: border-color 0.14s ease;
}

.panel-input::placeholder {
  color: var(--text-dim);
}

.panel-input:focus {
  border-color: rgba(var(--accent-rgb), 0.5);
}

/* ── Add version button ──────────────────────────────── */
.add-btn {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--text-dim);
  background: transparent;
  border: none;
  cursor: pointer;
  padding: 4px 2px;
  transition: color 0.14s ease;
}

.add-btn:hover {
  color: var(--accent);
}

/* ── List toolbar ────────────────────────────────────── */
.list-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 7px 10px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.list-count {
  font-size: 10.5px;
  color: var(--text-dim);
  font-family: ui-monospace, monospace;
}

/* ── List states ─────────────────────────────────────── */
.list-loading,
.list-empty {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  color: var(--text-dim);
}

.list-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 5px;
}

/* ── List item ───────────────────────────────────────── */
.list-item {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  border-radius: 7px;
  border: 1px solid var(--border-bright);
  padding: 8px 10px;
  background: var(--surface-2);
  transition: border-color 0.14s ease;
}

.list-item:hover {
  border-color: rgba(255, 255, 255, 0.14);
}

.list-item-body {
  flex: 1;
  min-width: 0;
}

.list-item-text {
  font-size: 11.5px;
  color: var(--text);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  word-break: break-words;
  margin: 0 0 3px;
  line-height: 1.5;
}

.list-item-meta {
  display: flex;
  gap: 8px;
  font-size: 10px;
  color: var(--text-dim);
}

.meta-tag {
  color: rgba(var(--accent-rgb), 0.78);
  font-family: ui-monospace, monospace;
}

.list-item-source {
  font-size: 10px;
  color: var(--text-dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin: 3px 0 0;
  font-family: ui-monospace, monospace;
}

/* ── Danger zone ─────────────────────────────────────── */
.danger-zone {
  flex-shrink: 0;
  border-top: 1px solid var(--border);
  padding: 8px 10px;
}

.danger-btn {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  font-size: 11px;
  padding: 6px;
  border-radius: 6px;
  border: 1px solid var(--border-bright);
  background: transparent;
  color: var(--text-dim);
  cursor: pointer;
  font-family: ui-monospace, monospace;
  letter-spacing: 0.04em;
  transition: all 0.14s ease;
}

.danger-btn:hover {
  color: var(--error);
  border-color: rgba(var(--error-rgb), 0.4);
  background: rgba(var(--error-rgb), 0.12);
}

.danger-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

/* ── Icon buttons ────────────────────────────────────── */
.icon-btn {
  width: 24px;
  height: 24px;
  border-radius: 5px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: 1px solid transparent;
  color: var(--text-dim);
  cursor: pointer;
  transition: all 0.13s ease;
}

.icon-btn:hover {
  background: var(--surface-3);
  border-color: var(--border-bright);
  color: var(--text);
}

.icon-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.icon-btn--accent:hover {
  color: var(--accent);
  border-color: rgba(var(--accent-rgb), 0.34);
  background: rgba(var(--accent-rgb), 0.14);
}

.icon-btn--danger:hover {
  color: var(--error);
  border-color: rgba(var(--error-rgb), 0.34);
  background: rgba(var(--error-rgb), 0.14);
}

/* ── Text buttons ────────────────────────────────────── */
.text-btn {
  font-size: 10.5px;
  padding: 2px 7px;
  border-radius: 5px;
  border: 1px solid var(--border-bright);
  background: transparent;
  color: var(--text-dim);
  cursor: pointer;
  transition: all 0.13s ease;
}

.text-btn--danger:hover {
  color: var(--error);
  border-color: rgba(var(--error-rgb), 0.4);
}

.text-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

/* ── Form buttons ────────────────────────────────────── */
.form-btn {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 11.5px;
  font-weight: 500;
  cursor: pointer;
  border: 1px solid var(--border-bright);
  transition: all 0.14s ease;
}

.form-btn--primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #0a0b0f;
}

.form-btn--primary:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.form-btn--ghost {
  background: transparent;
  color: var(--text-dim);
}

.form-btn--ghost:hover {
  color: var(--text);
  background: var(--surface-3);
}

.form-btn--error {
  background: var(--error);
  border-color: var(--error);
  color: #fff;
}

.form-btn--error:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* ── Overlay ─────────────────────────────────────────── */
.overlay {
  position: fixed;
  inset: 0;
  z-index: 50;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.65);
  backdrop-filter: blur(8px);
}

.confirm-dialog {
  background: var(--surface-2);
  border: 1px solid var(--border-bright);
  border-radius: 12px;
  padding: 20px;
  max-width: 300px;
  width: calc(100% - 32px);
  box-shadow: 0 24px 48px rgba(0, 0, 0, 0.5);
}

.confirm-header {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 16px;
}

.error-icon {
  color: var(--error);
  flex-shrink: 0;
  margin-top: 1px;
}

.confirm-msg {
  font-size: 13px;
  font-weight: 500;
  color: var(--text);
  margin: 0;
  line-height: 1.45;
}

.confirm-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

/* ── Shared utilities ────────────────────────────────── */
.spin {
  animation: spin 0.8s linear infinite;
}

.accent-icon {
  color: var(--accent);
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
