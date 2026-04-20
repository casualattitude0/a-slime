<script setup lang="ts">
import { ref, watch, nextTick, onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { Trash2, Loader2 } from 'lucide-vue-next'
import { useChatStore } from '../stores/chatStore'
import ChatMessage from './ChatMessage.vue'
import ChatInput from './ChatInput.vue'
import aiSlimeAvatar from '../assets/ai_slime_avatar.png'

const chatStore = useChatStore()
const { messages, status, isLoading } = storeToRefs(chatStore)
const logRef = ref<HTMLElement | null>(null)

const scrollToBottom = async () => {
  await nextTick()
  if (logRef.value) {
    logRef.value.scrollTop = logRef.value.scrollHeight
  }
}

watch(
  () => messages.value,
  () => {
    scrollToBottom()
  },
  { deep: true }
)

onMounted(() => {
  scrollToBottom()
})

const handleSend = (text: string) => {
  chatStore.sendMessage(text)
}

const handleClear = () => {
  if (confirm('Are you sure you want to clear the chat history?')) {
    chatStore.clearHistory()
  }
}
</script>

<template>
  <div class="flex h-screen max-w-6xl mx-auto px-4 py-6 gap-8">
    <!-- Main Chat Area -->
    <div class="flex-1 flex flex-col h-full min-w-0">
      <!-- Header -->
      <header class="flex items-center justify-between mb-6">
        <h1 class="text-xl font-semibold text-gray-100">Local Agent</h1>
        
        <button 
          @click="handleClear"
          class="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg text-gray-400 hover:text-error hover:bg-surface transition-colors"
          :disabled="isLoading"
        >
          <Trash2 :size="16" />
          Clear History
        </button>
      </header>
      
      <!-- Chat Log -->
      <main 
        ref="logRef"
        class="flex-1 overflow-y-auto mb-6 pr-2 scroll-smooth"
      >
        <div v-if="messages.length === 0" class="flex flex-col items-center justify-center h-full text-gray-500">
          <div class="w-16 h-16 bg-surface rounded-full flex items-center justify-center mb-4">
            <Loader2 class="animate-spin text-accent" :size="32" />
          </div>
          <p>Start a conversation...</p>
        </div>
        
        <ChatMessage 
          v-for="(msg, i) in messages" 
          :key="i"
          :role="msg.role"
          :text="msg.text"
        />
      </main>
      
      <!-- Input Area -->
      <footer class="shrink-0">
        <ChatInput 
          :disabled="isLoading"
          @send="handleSend"
        />
      </footer>
    </div>

    <!-- Avatar Sidebar -->
    <div class="flex flex-col w-64 items-center justify-end pb-8 shrink-0">
      <!-- Status Indicator -->
      <div v-if="status || isLoading" class="flex flex-col items-center gap-2 mb-4 text-sm text-accent animate-pulse">
        <Loader2 class="animate-spin" :size="24" />
        <span class="text-center">{{ status || 'Thinking...' }}</span>
      </div>
      
      <img :src="aiSlimeAvatar" alt="AI Agent Avatar" class="w-48 h-48 object-contain" />
    </div>
  </div>
</template>

<style>
/* Custom scrollbar for webkit */
::-webkit-scrollbar {
  width: 8px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: #373a40;
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background: #495057;
}
</style>
