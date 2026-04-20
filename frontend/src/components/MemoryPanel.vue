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
  <div class="flex flex-col h-full bg-surface rounded-xl border border-gray-700 overflow-hidden">
    <!-- Tab bar -->
    <div class="flex border-b border-gray-700 shrink-0">
      <button
        v-for="tab in (['versions', 'memory', 'rag'] as const)"
        :key="tab"
        @click="loadTab(tab)"
        class="flex-1 flex items-center justify-center gap-1.5 py-2.5 text-xs font-medium transition-colors"
        :class="activeTab === tab
          ? 'text-accent border-b-2 border-accent'
          : 'text-gray-500 hover:text-gray-300'"
      >
        <Layers v-if="tab === 'versions'" :size="13" />
        <Brain v-else-if="tab === 'memory'" :size="13" />
        <FileText v-else :size="13" />
        {{ tab === 'versions' ? '版本' : tab === 'memory' ? '記憶' : 'RAG' }}
      </button>
    </div>

    <!-- ── Versions tab ─────────────────────────────────────────────────── -->
    <div v-if="activeTab === 'versions'" class="flex-1 overflow-y-auto p-3 flex flex-col gap-2">
      <div
        v-for="v in versions"
        :key="v.version_id"
        class="rounded-lg border p-2.5 text-xs transition-colors"
        :class="v.is_active
          ? 'border-accent bg-accent/10'
          : 'border-gray-700 bg-gray-800/40 hover:border-gray-600'"
      >
        <div class="flex items-start justify-between gap-2">
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-1.5">
              <Check v-if="v.is_active" :size="12" class="text-accent shrink-0" />
              <span class="font-medium text-gray-200 truncate">{{ v.name }}</span>
            </div>
            <div class="text-gray-500 mt-0.5 truncate">{{ v.model_profile }}</div>
            <div class="text-gray-600 mt-0.5 truncate">{{ fmtDate(v.created_at) }}</div>
          </div>
          <div class="flex items-center gap-1 shrink-0">
            <button
              v-if="!v.is_active"
              @click="handleSwitchVersion(v.version_id)"
              :disabled="!!actionLoading"
              class="p-1 rounded text-gray-400 hover:text-accent hover:bg-gray-700 disabled:opacity-40 transition-colors"
              title="切換至此版本"
            >
              <Loader2 v-if="actionLoading === `switch-${v.version_id}`" :size="13" class="animate-spin" />
              <ChevronRight v-else :size="13" />
            </button>
            <button
              @click="handleDeleteVersion(v.version_id)"
              :disabled="!!actionLoading || (v.is_active && versions.length === 1)"
              class="p-1 rounded text-gray-500 hover:text-error hover:bg-gray-700 disabled:opacity-40 transition-colors"
              title="刪除版本"
            >
              <Loader2 v-if="actionLoading === `del-ver-${v.version_id}`" :size="13" class="animate-spin" />
              <Trash2 v-else :size="13" />
            </button>
          </div>
        </div>
      </div>

      <!-- New version form -->
      <div v-if="showNewVersionForm" class="rounded-lg border border-gray-700 p-2.5 text-xs flex flex-col gap-2">
        <input
          v-model="newVersionName"
          placeholder="版本名稱"
          @keyup.enter="handleCreateVersion"
          class="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-gray-200 placeholder-gray-600 focus:outline-none focus:border-accent text-xs"
        />
        <select
          v-model="newVersionProfile"
          class="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-gray-200 focus:outline-none focus:border-accent text-xs"
        >
          <option v-for="p in availableProfiles" :key="p" :value="p">{{ p }}</option>
        </select>
        <div class="flex gap-1.5">
          <button
            @click="handleCreateVersion"
            :disabled="!newVersionName.trim() || !!actionLoading"
            class="flex-1 flex items-center justify-center gap-1 py-1.5 rounded bg-accent text-gray-900 font-medium disabled:opacity-40 hover:bg-accent/90 transition-colors"
          >
            <Loader2 v-if="actionLoading === 'create'" :size="12" class="animate-spin" />
            <span>建立</span>
          </button>
          <button
            @click="showNewVersionForm = false"
            class="px-3 py-1.5 rounded border border-gray-700 text-gray-400 hover:text-gray-200 transition-colors"
          >
            <X :size="12" />
          </button>
        </div>
      </div>

      <button
        v-if="!showNewVersionForm"
        @click="showNewVersionForm = true"
        class="flex items-center gap-1.5 text-xs text-gray-500 hover:text-accent transition-colors mt-1"
      >
        <Plus :size="12" />
        新增版本
      </button>
    </div>

    <!-- ── Memory tab ───────────────────────────────────────────────────── -->
    <div v-else-if="activeTab === 'memory'" class="flex-1 overflow-y-auto flex flex-col">
      <div class="flex items-center justify-between px-3 py-2 border-b border-gray-700/60 shrink-0">
        <span class="text-xs text-gray-500">{{ memoryItems.length }} 筆記憶</span>
        <div class="flex gap-1.5">
          <button @click="chatStore.fetchMemoryItems()" :disabled="memoryLoading" class="p-1 rounded text-gray-500 hover:text-gray-300 disabled:opacity-40 transition-colors">
            <RefreshCw :size="12" :class="memoryLoading ? 'animate-spin' : ''" />
          </button>
          <button
            v-if="memoryItems.length > 0"
            @click="confirmDeleteAll = 'memory'"
            :disabled="!!actionLoading"
            class="text-xs px-2 py-0.5 rounded border border-gray-700 text-gray-500 hover:text-error hover:border-error disabled:opacity-40 transition-colors"
          >清空</button>
        </div>
      </div>

      <div v-if="memoryLoading" class="flex-1 flex items-center justify-center">
        <Loader2 :size="20" class="animate-spin text-accent" />
      </div>
      <div v-else-if="memoryItems.length === 0" class="flex-1 flex items-center justify-center text-xs text-gray-600">無記憶項目</div>
      <div v-else class="flex-1 overflow-y-auto p-2 flex flex-col gap-1.5">
        <div
          v-for="item in memoryItems"
          :key="item.id"
          class="rounded-lg border border-gray-700 p-2 text-xs flex items-start gap-2 hover:border-gray-600 bg-gray-800/40"
        >
          <div class="flex-1 min-w-0">
            <p class="text-gray-300 line-clamp-2 break-words">{{ item.content }}</p>
            <div class="flex gap-2 mt-1 text-gray-600">
              <span v-if="item.metadata?.ts">{{ fmtDate(item.metadata.ts) }}</span>
              <span v-if="item.metadata?.tags" class="text-accent/70">{{ item.metadata.tags }}</span>
            </div>
          </div>
          <button
            @click="handleDeleteMemory(item.id)"
            :disabled="!!actionLoading"
            class="shrink-0 p-1 rounded text-gray-600 hover:text-error hover:bg-gray-700 disabled:opacity-40 transition-colors"
          >
            <Loader2 v-if="actionLoading === `del-mem-${item.id}`" :size="12" class="animate-spin" />
            <Trash2 v-else :size="12" />
          </button>
        </div>
      </div>
    </div>

    <!-- ── RAG tab ──────────────────────────────────────────────────────── -->
    <div v-else-if="activeTab === 'rag'" class="flex-1 overflow-y-auto flex flex-col">
      <div class="flex items-center justify-between px-3 py-2 border-b border-gray-700/60 shrink-0">
        <span class="text-xs text-gray-500">{{ ragItems.length }} 筆文件</span>
        <div class="flex gap-1.5">
          <button @click="chatStore.fetchRagItems()" :disabled="ragLoading" class="p-1 rounded text-gray-500 hover:text-gray-300 disabled:opacity-40 transition-colors">
            <RefreshCw :size="12" :class="ragLoading ? 'animate-spin' : ''" />
          </button>
          <button
            v-if="ragItems.length > 0"
            @click="confirmDeleteAll = 'rag'"
            :disabled="!!actionLoading"
            class="text-xs px-2 py-0.5 rounded border border-gray-700 text-gray-500 hover:text-error hover:border-error disabled:opacity-40 transition-colors"
          >清空</button>
        </div>
      </div>

      <div v-if="ragLoading" class="flex-1 flex items-center justify-center">
        <Loader2 :size="20" class="animate-spin text-accent" />
      </div>
      <div v-else-if="ragItems.length === 0" class="flex-1 flex items-center justify-center text-xs text-gray-600">無 RAG 文件</div>
      <div v-else class="flex-1 overflow-y-auto p-2 flex flex-col gap-1.5">
        <div
          v-for="item in ragItems"
          :key="item.id"
          class="rounded-lg border border-gray-700 p-2 text-xs flex items-start gap-2 hover:border-gray-600 bg-gray-800/40"
        >
          <div class="flex-1 min-w-0">
            <p class="text-gray-300 line-clamp-2 break-words">{{ item.content }}</p>
            <p v-if="item.metadata?.source" class="text-gray-600 mt-0.5 truncate">{{ item.metadata.source }}</p>
          </div>
          <button
            @click="handleDeleteRag(item.id)"
            :disabled="!!actionLoading"
            class="shrink-0 p-1 rounded text-gray-600 hover:text-error hover:bg-gray-700 disabled:opacity-40 transition-colors"
          >
            <Loader2 v-if="actionLoading === `del-rag-${item.id}`" :size="12" class="animate-spin" />
            <Trash2 v-else :size="12" />
          </button>
        </div>
      </div>
    </div>

    <!-- ── Danger zone ─────────────────────────────────────────────────── -->
    <div class="shrink-0 border-t border-gray-700/60 p-2">
      <button
        @click="confirmDeleteAll = 'all'"
        :disabled="!!actionLoading || isLoading"
        class="w-full text-xs py-1.5 rounded border border-gray-700 text-gray-600 hover:text-error hover:border-error disabled:opacity-40 transition-colors"
      >
        <AlertTriangle class="inline mr-1" :size="11" />
        全部清除
      </button>
    </div>

    <!-- ── Confirm overlay ─────────────────────────────────────────────── -->
    <Teleport to="body">
      <div
        v-if="confirmDeleteAll"
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
        @click.self="confirmDeleteAll = ''"
      >
        <div class="bg-gray-900 border border-gray-700 rounded-xl p-5 max-w-xs w-full mx-4 shadow-2xl">
          <div class="flex items-center gap-2 mb-3">
            <AlertTriangle :size="18" class="text-error shrink-0" />
            <p class="text-sm font-medium text-gray-200">
              {{ confirmDeleteAll === 'all'
                ? '確認全部清除？此操作不可復原。'
                : confirmDeleteAll === 'memory'
                  ? '確認清空所有記憶？'
                  : '確認清空所有 RAG 文件？' }}
            </p>
          </div>
          <div class="flex gap-2 justify-end">
            <button
              @click="confirmDeleteAll = ''"
              class="px-3 py-1.5 text-xs rounded border border-gray-700 text-gray-400 hover:text-gray-200 transition-colors"
            >取消</button>
            <button
              @click="
                confirmDeleteAll === 'all'
                  ? handleDeleteAll()
                  : confirmDeleteAll === 'memory'
                    ? handleDeleteAllMemory()
                    : handleDeleteAllRag()
              "
              :disabled="!!actionLoading"
              class="px-3 py-1.5 text-xs rounded bg-error text-white font-medium disabled:opacity-60 hover:bg-error/80 transition-colors"
            >
              <Loader2
                v-if="actionLoading === 'del-all' || actionLoading === 'del-mem-all' || actionLoading === 'del-rag-all'"
                :size="12"
                class="animate-spin inline mr-1"
              />
              確認刪除
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>
