"use client"

import {
  SPARK_PITCH_LEVELS,
  SPARK_SPEED_LEVELS,
  SPARK_LEVEL_LABELS,
  type SparkVoiceSettings,
} from "@/lib/voice-types"

interface SparkVoiceSettingsProps {
  settings: SparkVoiceSettings
  onChange: (s: SparkVoiceSettings) => void
}

function PillGroup<T extends string>({
  options,
  value,
  getLabel,
  onChange,
}: {
  options: readonly T[]
  value: T
  getLabel: (v: T) => string
  onChange: (v: T) => void
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {options.map((opt) => (
        <button
          key={opt}
          type="button"
          onClick={() => onChange(opt)}
          className={`px-2 py-0.5 text-xs rounded-md ${
            value === opt
              ? "bg-sky-100 dark:bg-sky-900 text-sky-600 dark:text-sky-300"
              : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"
          }`}
        >
          {getLabel(opt)}
        </button>
      ))}
    </div>
  )
}

export function SparkVoiceSettings({ settings, onChange }: SparkVoiceSettingsProps) {
  const cloneReady =
    settings.mode === "clone" &&
    settings.refAudioPath.trim() !== "" &&
    settings.refText.trim() !== ""

  const cloneMissing =
    settings.mode === "clone" &&
    (settings.refAudioPath.trim() === "" || settings.refText.trim() === "")

  return (
    <div className="space-y-3">
      {/* Mode switcher */}
      <div className="flex rounded-md border border-gray-200 dark:border-gray-700 overflow-hidden text-xs">
        <button
          type="button"
          onClick={() => onChange({ ...settings, mode: "create" })}
          className={`flex-1 py-1.5 font-medium transition-colors ${
            settings.mode === "create"
              ? "bg-sky-500 text-white"
              : "bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
          }`}
        >
          Create voice
        </button>
        <button
          type="button"
          onClick={() => onChange({ ...settings, mode: "clone" })}
          className={`flex-1 py-1.5 font-medium transition-colors ${
            settings.mode === "clone"
              ? "bg-sky-500 text-white"
              : "bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
          }`}
        >
          Clone voice
        </button>
      </div>

      {settings.mode === "create" && (
        <>
          <div>
            <span className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Gender</span>
            <PillGroup
              options={["male", "female"] as const}
              value={settings.gender}
              getLabel={(v) => v.charAt(0).toUpperCase() + v.slice(1)}
              onChange={(v) => onChange({ ...settings, gender: v })}
            />
          </div>

          <div>
            <span className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Pitch</span>
            <PillGroup
              options={SPARK_PITCH_LEVELS}
              value={settings.pitchLevel}
              getLabel={(v) => SPARK_LEVEL_LABELS[v]}
              onChange={(v) => onChange({ ...settings, pitchLevel: v })}
            />
          </div>

          <div>
            <span className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Speed</span>
            <PillGroup
              options={SPARK_SPEED_LEVELS}
              value={settings.speedLevel}
              getLabel={(v) => SPARK_LEVEL_LABELS[v]}
              onChange={(v) => onChange({ ...settings, speedLevel: v })}
            />
          </div>

          <p className="text-xs text-gray-400 dark:text-gray-500">
            Spark generates a new voice from these characteristics. No predefined voice IDs exist.
          </p>
        </>
      )}

      {settings.mode === "clone" && (
        <div className="space-y-2">
          {cloneMissing && (
            <div className="rounded-md bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 px-3 py-2 text-xs text-amber-700 dark:text-amber-400">
              Both reference audio and transcript are required for voice cloning.
            </div>
          )}
          {cloneReady && (
            <div className="inline-flex items-center gap-1 rounded-full bg-sky-100 dark:bg-sky-900 px-2 py-0.5 text-xs text-sky-700 dark:text-sky-300">
              <span className="h-1.5 w-1.5 rounded-full bg-sky-500"></span>
              Ready to clone
            </div>
          )}
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
  )
}
