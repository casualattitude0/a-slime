/** User-facing label for version `model_profile` (API value stays lowercase slug). */
const LABELS: Record<string, string> = {
  default: '預設',
  gemini: 'Gemini',
  'gemini-pro': 'Gemini Pro',
  ollama: 'Ollama',
  nvidia: 'NVIDIA',
}

export function formatModelProfileLabel(profile: string): string {
  return LABELS[profile] ?? profile
}
