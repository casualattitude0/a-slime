import { watch, nextTick, type Ref } from 'vue'

export function useHeroToChatBubbleFly(opts: {
  streamingBotIndex: Ref<number>
  streamingReply: Ref<string>
  logRef: Ref<HTMLElement | null>
  avatarRef: Ref<HTMLElement | null>
  /** ms to keep the bubble anchored above the avatar before flying into the transcript */
  holdMs?: number
  /** px upward from avatar vertical center for the hold position */
  holdAboveAvatarPx?: number
  /** Called when the fly finishes or is skipped — reveal chat bubble */
  onFlyArrived?: () => void
}) {
  const holdMs = opts.holdMs ?? 520
  const holdAboveAvatarPx = opts.holdAboveAvatarPx ?? 56

  watch(
    () => opts.streamingBotIndex.value,
    async (idx, prevIdx) => {
      if (idx < 0 || prevIdx !== -1) return
      await nextTick()
      await nextTick()

      requestAnimationFrame(() => {
        const arrive = () => {
          opts.onFlyArrived?.()
        }

        const reducedMotion =
          typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches

        const avatar = opts.avatarRef.value
        const scrollRoot = opts.logRef.value
        const target = scrollRoot?.querySelector('[data-bot-streaming]') as HTMLElement | null

        if (reducedMotion || !avatar || !scrollRoot || !target) {
          arrive()
          return
        }

        try {
          const sr = avatar.getBoundingClientRect()
          const er = target.getBoundingClientRect()
          const scx = sr.left + sr.width / 2
          const scy = sr.top + sr.height / 2 - holdAboveAvatarPx
          const ecx = er.left + er.width / 2
          const ecy = er.top + er.height / 2
          const ox = scx - ecx
          const oy = scy - ecy

          const fly = document.createElement('div')
          fly.className = 'bubble-fly-clone'
          fly.textContent = opts.streamingReply.value.length ? opts.streamingReply.value : '…'
          fly.setAttribute('aria-hidden', 'true')
          document.body.appendChild(fly)

          const fx = ecx
          const fy = ecy
          fly.style.left = `${fx}px`
          fly.style.top = `${fy}px`
          fly.style.transition = 'none'
          fly.style.transform = `translate(-50%, -50%) translate(${ox}px, ${oy}px) scale(0.82)`
          fly.style.opacity = '0.92'

          let cleaned = false
          const stopTextWatch = watch(
            () => opts.streamingReply.value,
            (t) => {
              if (cleaned || !fly.parentNode) return
              fly.textContent = t.length ? t : '…'
            },
            { flush: 'sync' },
          )

          const cleanup = () => {
            if (cleaned) return
            cleaned = true
            stopTextWatch()
            fly.remove()
          }

          const flyDurationMs = 580

          let flySettled = false
          const finishFly = () => {
            if (flySettled) return
            flySettled = true
            cleanup()
            arrive()
          }

          window.setTimeout(() => {
            // Animate transform only — two properties would fire multiple transitionend + { once: true } can swallow the wrong one.
            fly.style.transition = `transform ${flyDurationMs / 1000}s cubic-bezier(0.22, 1, 0.36, 1)`
            requestAnimationFrame(() => {
              fly.style.transform = 'translate(-50%, -50%) translate(0px, 0px) scale(1)'
              fly.style.opacity = '1'
            })

            fly.addEventListener(
              'transitionend',
              (e) => {
                if (e.propertyName !== 'transform') return
                finishFly()
              },
              { once: true },
            )

            window.setTimeout(finishFly, flyDurationMs + 420)
          }, holdMs)
        } catch {
          arrive()
        }
      })
    },
  )
}
