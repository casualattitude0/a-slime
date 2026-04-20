<script setup lang="ts">
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'
import { User, AlertTriangle } from 'lucide-vue-next'
import aiSlimeAvatar from '../assets/ai_slime_avatar.png'

const props = defineProps<{
  role: 'user' | 'bot' | 'err'
  text: string
}>()

const md = new MarkdownIt({
  breaks: true,
  linkify: true,
})

const renderedText = computed(() => {
  if (props.role === 'bot') {
    return md.render(props.text)
  }
  return props.text
})
</script>

<template>
  <div 
    class="flex gap-4 p-4 rounded-xl mb-4"
    :class="{
      'bg-surface': role === 'bot' || role === 'err',
      'bg-transparent': role === 'user'
    }"
  >
    <div class="shrink-0 mt-1">
      <div
        class="w-8 h-8 rounded-full flex items-center justify-center overflow-hidden"
        :class="{
          'bg-accent text-gray-900': role === 'user',
          'bg-gray-700 text-gray-200': role === 'bot',
          'bg-error text-white': role === 'err'
        }"
      >
        <User v-if="role === 'user'" :size="18" />
        <img
          v-else-if="role === 'bot'"
          :src="aiSlimeAvatar"
          alt="AI avatar"
          class="w-full h-full object-cover"
        />
        <AlertTriangle v-else :size="18" />
      </div>
    </div>
    
    <div class="flex-1 min-w-0">
      <div class="font-semibold mb-1 text-sm text-gray-400">
        {{ role === 'user' ? 'You' : role === 'bot' ? 'Agent' : 'Error' }}
      </div>
      
      <div 
        v-if="role === 'bot'" 
        class="prose prose-invert max-w-none text-gray-200"
        v-html="renderedText"
      ></div>
      <div 
        v-else-if="role === 'err'" 
        class="text-error whitespace-pre-wrap break-words text-sm"
      >
        {{ text }}
      </div>
      <div 
        v-else 
        class="text-gray-200 whitespace-pre-wrap break-words"
      >
        {{ text }}
      </div>
    </div>
  </div>
</template>

<style>
/* Basic prose styles for markdown-it output */
.prose p {
  margin-top: 0.5em;
  margin-bottom: 0.5em;
}
.prose pre {
  background-color: #1a1b1e;
  padding: 1em;
  border-radius: 0.5rem;
  overflow-x: auto;
  border: 1px solid #373a40;
}
.prose code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.875em;
  background-color: #1a1b1e;
  padding: 0.2em 0.4em;
  border-radius: 0.25rem;
}
.prose pre code {
  background-color: transparent;
  padding: 0;
}
.prose ul {
  list-style-type: disc;
  padding-left: 1.5em;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
}
.prose ol {
  list-style-type: decimal;
  padding-left: 1.5em;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
}
</style>
