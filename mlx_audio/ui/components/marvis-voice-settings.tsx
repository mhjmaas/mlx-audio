"use client"

import { useState } from "react"
import { ChevronDown } from "lucide-react"
import type { MarvisVoiceSettings } from "@/lib/voice-types"

interface MarvisVoiceSettingsProps {
  settings: MarvisVoiceSettings
  onChange: (s: MarvisVoiceSettings) => void
}

export function MarvisVoiceSettings({ settings, onChange }: MarvisVoiceSettingsProps) {
  const [cloneOpen, setCloneOpen] = useState(false)
  const isCloning = settings.refAudioPath.trim() !== "" && settings.refText.trim() !== ""

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500 dark:text-gray-400">Voice preset</span>
        <div className="relative">
          <select
            className="appearance-none rounded-md border border-gray-200 dark:border-gray-700 px-2 py-1 text-sm pr-7 bg-white dark:bg-gray-800"
            value={settings.voice}
            onChange={(e) =>
              onChange({ ...settings, voice: e.target.value as MarvisVoiceSettings["voice"] })
            }
          >
            <option value="conversational_a">Conversational A</option>
            <option value="conversational_b">Conversational B</option>
          </select>
          <ChevronDown className="absolute right-1.5 top-1.5 h-3.5 w-3.5 pointer-events-none text-gray-400" />
        </div>
      </div>

      {isCloning && (
        <div className="inline-flex items-center gap-1 rounded-full bg-sky-100 dark:bg-sky-900 px-2 py-0.5 text-xs text-sky-700 dark:text-sky-300">
          <span className="h-1.5 w-1.5 rounded-full bg-sky-500"></span>
          Using cloned voice
        </div>
      )}

      <div>
        <button
          type="button"
          className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
          onClick={() => setCloneOpen((v) => !v)}
        >
          <ChevronDown
            className={`h-3.5 w-3.5 transition-transform ${cloneOpen ? "rotate-180" : ""}`}
          />
          Voice cloning (optional)
        </button>

        {cloneOpen && (
          <div className="mt-2 space-y-2 rounded-md border border-gray-200 dark:border-gray-700 p-3">
            <div>
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                Reference audio path
              </label>
              <input
                type="text"
                className="w-full rounded-md border border-gray-200 dark:border-gray-700 px-2 py-1 text-xs bg-white dark:bg-gray-800 focus:border-blue-500 focus:outline-none"
                placeholder="/path/on/server/voice.wav"
                value={settings.refAudioPath}
                onChange={(e) => onChange({ ...settings, refAudioPath: e.target.value })}
              />
              <p className="mt-0.5 text-xs text-gray-400 dark:text-gray-500">
                File path accessible to the server process
              </p>
            </div>
            <div>
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                Reference transcript
              </label>
              <textarea
                className="w-full rounded-md border border-gray-200 dark:border-gray-700 px-2 py-1 text-xs bg-white dark:bg-gray-800 focus:border-blue-500 focus:outline-none"
                placeholder="Exact words spoken in the reference audio"
                rows={2}
                value={settings.refText}
                onChange={(e) => onChange({ ...settings, refText: e.target.value })}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
