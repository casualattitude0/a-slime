<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'
import { SendHorizontal } from 'lucide-vue-next'

const props = defineProps<{
  disabled: boolean
}>()

const emit = defineEmits<{
  (e: 'send', text: string): void
}>()

const input = ref('')
const textareaRef = ref<HTMLTextAreaElement | null>(null)
const isComposingWithIME = ref(false)

const adjustHeight = () => {
  if (!textareaRef.value) return
  textareaRef.value.style.height = 'auto'
  textareaRef.value.style.height = `${Math.min(textareaRef.value.scrollHeight, 200)}px`
}

const handleKeydown = (e: KeyboardEvent) => {
  const stillComposing = isComposingWithIME.value || e.isComposing || e.keyCode === 229
  const isCmdEnterOnly = e.key === 'Enter' && e.metaKey && !e.ctrlKey && !e.altKey && !e.shiftKey
  if (isCmdEnterOnly && !stillComposing) {
    e.preventDefault()
    send()
  }
}

const handleCompositionStart = () => {
  isComposingWithIME.value = true
}

const handleCompositionEnd = () => {
  isComposingWithIME.value = false
}

const send = () => {
  const text = input.value.trim()
  if (!text || props.disabled) return
  
  emit('send', text)
  input.value = ''
  
  nextTick(() => {
    adjustHeight()
    textareaRef.value?.focus()
  })
}

onMounted(() => {
  textareaRef.value?.focus()
})
</script>

<template>
  <div class="relative bg-surface rounded-xl border border-gray-700 shadow-sm focus-within:border-accent transition-colors">
    <textarea
      ref="textareaRef"
      v-model="input"
      class="w-full bg-transparent text-gray-100 placeholder-gray-500 border-none focus:ring-0 resize-none py-3 pl-4 pr-12 min-h-[52px] max-h-[200px] overflow-y-auto outline-none"
      placeholder="Message Agent..."
      :disabled="disabled"
      @input="adjustHeight"
      @keydown="handleKeydown"
      @compositionstart="handleCompositionStart"
      @compositionend="handleCompositionEnd"
      rows="1"
    ></textarea>
    
    <button
      @click="send"
      :disabled="!input.trim() || disabled"
      class="absolute right-2 bottom-2 p-2 rounded-lg text-gray-400 hover:text-accent hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
    >
      <SendHorizontal :size="20" />
    </button>
  </div>
</template>
