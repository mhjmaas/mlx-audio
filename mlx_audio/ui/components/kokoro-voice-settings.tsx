"use client"

import { ChevronDown } from "lucide-react"
import {
  KOKORO_VOICE_GROUPS,
  KOKORO_VOICE_BY_ID,
  KOKORO_LANG_CODES,
} from "@/lib/kokoro-voices"
import type { KokoroVoiceSettings } from "@/lib/voice-types"

interface KokoroVoiceSettingsProps {
  settings: KokoroVoiceSettings
  onChange: (s: KokoroVoiceSettings) => void
}

export function KokoroVoiceSettings({ settings, onChange }: KokoroVoiceSettingsProps) {
  const isBlending = settings.voice.includes(",")

  const handleVoiceSelect = (voiceId: string) => {
    const meta = KOKORO_VOICE_BY_ID[voiceId]
    onChange({
      voice: voiceId,
      langCode: meta ? meta.langCode : settings.langCode,
    })
  }

  const handleBlendChange = (value: string) => {
    // When user manually types a blend, keep the current langCode
    onChange({ ...settings, voice: value })
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500 dark:text-gray-400">Voice</span>
        <div className="relative">
          <select
            className="appearance-none rounded-md border border-gray-200 dark:border-gray-700 px-2 py-1 text-sm pr-7 bg-white dark:bg-gray-800 max-w-[160px]"
            value={isBlending ? "" : settings.voice}
            onChange={(e) => handleVoiceSelect(e.target.value)}
          >
            {isBlending && (
              <option value="" disabled>
                (blended)
              </option>
            )}
            {KOKORO_VOICE_GROUPS.map((group) => (
              <optgroup key={group.label} label={group.label}>
                {group.voices.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.label}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
          <ChevronDown className="absolute right-1.5 top-1.5 h-3.5 w-3.5 pointer-events-none text-gray-400" />
        </div>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500 dark:text-gray-400">Language / Accent</span>
        <div className="relative">
          <select
            className="appearance-none rounded-md border border-gray-200 dark:border-gray-700 px-2 py-1 text-sm pr-7 bg-white dark:bg-gray-800 max-w-[160px]"
            value={settings.langCode}
            onChange={(e) => onChange({ ...settings, langCode: e.target.value })}
          >
            {KOKORO_LANG_CODES.map((lc) => (
              <option key={lc.code} value={lc.code}>
                {lc.label}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-1.5 top-1.5 h-3.5 w-3.5 pointer-events-none text-gray-400" />
        </div>
      </div>

      <div>
        <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
          Blend voices{" "}
          <span className="text-gray-400 dark:text-gray-500">(comma-separated IDs)</span>
        </label>
        <input
          type="text"
          className="w-full rounded-md border border-gray-200 dark:border-gray-700 px-2 py-1 text-xs bg-white dark:bg-gray-800 focus:border-blue-500 focus:outline-none font-mono"
          placeholder="af_heart,am_adam"
          value={settings.voice}
          onChange={(e) => handleBlendChange(e.target.value)}
        />
        <p className="mt-0.5 text-xs text-gray-400 dark:text-gray-500">
          Multiple IDs average the voice embeddings. Single ID uses the picker above.
        </p>
      </div>
    </div>
  )
}
