<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import ChatLayout from './components/ChatLayout.vue'
import ChatAgentSim from './pages/ChatAgentSim.vue'
import MessageDisplayPlayground from './pages/MessageDisplayPlayground.vue'

const PLAYGROUND_HASH = '#/message-playground'
const CHAT_SIM_HASH = '#/chat-sim'

type DevRoute = 'home' | 'message-playground' | 'chat-sim'

function routeFromHash(): DevRoute {
  const h = window.location.hash
  if (h === PLAYGROUND_HASH) return 'message-playground'
  if (h === CHAT_SIM_HASH) return 'chat-sim'
  return 'home'
}

const route = ref<DevRoute>(routeFromHash())

function onHashChange() {
  route.value = routeFromHash()
}

onMounted(() => window.addEventListener('hashchange', onHashChange))
onUnmounted(() => window.removeEventListener('hashchange', onHashChange))
</script>

<template>
  <MessageDisplayPlayground v-if="route === 'message-playground'" />
  <ChatAgentSim v-else-if="route === 'chat-sim'" />
  <ChatLayout v-else />
</template>
