"use client"

import type React from "react"

import { useState, useRef } from "react"
import { ChevronDown, Download, ThumbsUp, ThumbsDown, Play, Pause, RefreshCw } from "lucide-react"
import { LayoutWrapper } from "@/components/layout-wrapper"
import { MarvisVoiceSettings } from "@/components/marvis-voice-settings"
import { KokoroVoiceSettings } from "@/components/kokoro-voice-settings"
import { SparkVoiceSettings } from "@/components/spark-voice-settings"
import {
  type VoiceSettings,
  getModelFamily,
  DEFAULT_MARVIS,
  DEFAULT_KOKORO,
  DEFAULT_SPARK,
  DEFAULT_QWEN3,
  SPARK_PITCH_MAP,
  SPARK_SPEED_MAP,
} from "@/lib/voice-types"

// Custom range input component with colored progress
function RangeInput({
  min,
  max,
  step,
  value,
  onChange,
  className = "",
  ariaLabel,
}: {
  min: number
  max: number
  step: number
  value: number
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  className?: string
  ariaLabel?: string
}) {
  const percentage = ((value - Number(min)) / (Number(max) - Number(min))) * 100

  return (
    <div className={`relative flex-1 ${className}`}>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={onChange}
        className="w-full appearance-none bg-transparent cursor-pointer"
        aria-label={ariaLabel}
        style={{
          background: `linear-gradient(to right, #0ea5e9 0%, #0ea5e9 ${percentage}%, #e5e7eb ${percentage}%, #e5e7eb 100%)`,
          height: "2px",
          borderRadius: "2px",
        }}
      />
    </div>
  )
}

export default function SpeechSynthesis() {
  const [text, setText] = useState("But I also have other interests, such as playing tic-tac-toe.")
  const [isPlaying, setIsPlaying] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [pitch, setPitch] = useState(0)
  const [volume, setVolume] = useState(1)
  const [currentTime, setCurrentTime] = useState("00:00")
  const [duration, setDuration] = useState("00:04")
  const [activeTab, setActiveTab] = useState<"settings" | "history">("settings")
  const [baseModel, setBaseModel] = useState("Marvis-AI/marvis-tts-100m-v0.2")
  const [quantization, setQuantization] = useState("6bit")
  const [language, setLanguage] = useState("English-detected")
  const [liked, setLiked] = useState<boolean | null>(null)
  const [voiceSettings, setVoiceSettings] = useState<VoiceSettings>({
    family: "marvis",
    ...DEFAULT_MARVIS,
  })

  const audioRef = useRef<HTMLAudioElement | null>(null)

  const isMarvisModel = (modelName: string) => modelName.toLowerCase().includes("marvis")

  const getAvailableQuantizations = (modelName: string) => {
    if (modelName === "Marvis-AI/marvis-tts-100m-v0.2") return ["none", "6bit", "8bit"]
    if (modelName === "Marvis-AI/marvis-tts-250m-v0.2") return ["none", "4bit", "6bit", "8bit"]
    if (modelName === "Marvis-AI/marvis-tts-250m-v0.1") return ["none", "4bit", "8bit"]
    return []
  }

  const getFullModelName = () => {
    if (isMarvisModel(baseModel) && quantization !== "none") {
      return `${baseModel}-MLX-${quantization}`
    }
    return baseModel
  }

  const model = getFullModelName()

  const showPitchSlider =
    voiceSettings.family === "marvis" || voiceSettings.family === "kokoro"

  const isGenerateDisabled =
    isGenerating ||
    (voiceSettings.family === "spark" &&
      voiceSettings.mode === "clone" &&
      (!voiceSettings.refAudioPath.trim() || !voiceSettings.refText.trim()))

  const voiceDisplayLabel = (() => {
    switch (voiceSettings.family) {
      case "marvis":
        return voiceSettings.refAudioPath ? "Cloned voice" : voiceSettings.voice
      case "kokoro":
        return voiceSettings.voice
      case "spark":
        return voiceSettings.mode === "clone"
          ? "Cloned voice"
          : `${voiceSettings.gender} / ${voiceSettings.pitchLevel} pitch`
      case "qwen3":
        return "Custom voice"
    }
  })()

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
  }

  const handleModelChange = (newModel: string) => {
    setBaseModel(newModel)
    if (isMarvisModel(newModel)) {
      const availableQuants = getAvailableQuantizations(newModel)
      if (availableQuants.includes("6bit")) {
        setQuantization("6bit")
      } else if (availableQuants.length > 0) {
        setQuantization(availableQuants[0])
      }
    }
    const family = getModelFamily(newModel)
    switch (family) {
      case "marvis": setVoiceSettings({ family: "marvis", ...DEFAULT_MARVIS }); break
      case "kokoro": setVoiceSettings({ family: "kokoro", ...DEFAULT_KOKORO }); break
      case "spark":  setVoiceSettings({ family: "spark",  ...DEFAULT_SPARK  }); break
      case "qwen3":  setVoiceSettings({ family: "qwen3",  ...DEFAULT_QWEN3  }); break
    }
  }

  const handlePlayPause = () => {
    if (!audioRef.current || !audioRef.current.src || audioRef.current.src === window.location.href) {
      handleGenerate()
      return
    }
    if (isPlaying) {
      audioRef.current?.pause()
    } else {
      audioRef.current?.play()
    }
    setIsPlaying(!isPlaying)
  }

  const handleGenerate = async () => {
    if (!audioRef.current) return
    setIsGenerating(true)

    const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost"
    const API_PORT = process.env.NEXT_PUBLIC_API_PORT || "8000"

    try {
      const requestBody: Record<string, unknown> = {
        model: model,
        input: text,
        speed: speed,
      }

      switch (voiceSettings.family) {
        case "marvis":
          requestBody.voice = voiceSettings.voice
          requestBody.pitch = pitch
          if (voiceSettings.refAudioPath.trim()) {
            requestBody.ref_audio = voiceSettings.refAudioPath.trim()
            requestBody.ref_text = voiceSettings.refText
          }
          break

        case "kokoro":
          requestBody.voice = voiceSettings.voice
          requestBody.lang_code = voiceSettings.langCode
          requestBody.pitch = pitch
          break

        case "spark":
          if (voiceSettings.mode === "clone") {
            requestBody.ref_audio = voiceSettings.refAudioPath.trim()
            requestBody.ref_text = voiceSettings.refText
          } else {
            requestBody.gender = voiceSettings.gender
            requestBody.pitch = SPARK_PITCH_MAP[voiceSettings.pitchLevel]
            requestBody.speed = SPARK_SPEED_MAP[voiceSettings.speedLevel]
          }
          break

        case "qwen3": {
          requestBody.instruct = voiceSettings.instruct
          if (voiceSettings.useFixedSeed) {
            if (voiceSettings.seed.trim()) {
              requestBody.seed = parseInt(voiceSettings.seed, 10)
            } else {
              let hash = 0
              for (let i = 0; i < voiceSettings.instruct.length; i++) {
                const char = voiceSettings.instruct.charCodeAt(i)
                hash = ((hash << 5) - hash) + char
                hash = hash & hash
              }
              requestBody.seed = Math.abs(hash)
            }
          }
          break
        }
      }

      const response = await fetch(`${API_BASE_URL}:${API_PORT}/v1/audio/speech`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const blob = await response.blob()
      const audioUrl = URL.createObjectURL(blob)
      audioRef.current.src = audioUrl

      audioRef.current.onloadedmetadata = () => {
        setDuration(formatTime(Math.floor(audioRef.current?.duration || 0)))
        setCurrentTime("00:00")
        setIsPlaying(true)
        audioRef.current?.play()
      }

      audioRef.current.ontimeupdate = () => {
        setCurrentTime(formatTime(Math.floor(audioRef.current?.currentTime || 0)))
      }

      audioRef.current.onended = () => {
        setIsPlaying(false)
        setCurrentTime("00:00")
        if (audioRef.current?.src.startsWith("blob:")) {
          URL.revokeObjectURL(audioRef.current.src)
        }
      }
    } catch (error) {
      console.error("Error generating speech:", error)
    } finally {
      setIsGenerating(false)
    }
  }

  const handleDownload = () => {
    alert("Downloading audio...")
  }

  const handleFeedback = (isPositive: boolean) => {
    setLiked(isPositive)
  }

  const getCharacterCount = () => text.length

  const formatTime = (seconds: number) => `00:${seconds.toString().padStart(2, "0")}`

  return (
    <LayoutWrapper activeTab="audio" activePage="text-to-speech">
      <div className="flex flex-1 overflow-hidden">
        {/* Text Input Area */}
        <div className="flex-1 overflow-auto border-r border-gray-200 dark:border-gray-700 p-6">
          <h1 className="mb-6 text-2xl font-bold">Speech Synthesis</h1>
          <textarea
            className="min-h-[200px] w-full resize-none rounded-md border border-gray-200 dark:border-gray-700 p-4 text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 focus:border-blue-500 focus:outline-none"
            value={text}
            onChange={handleTextChange}
            placeholder="Enter text to convert to speech..."
          />
          <div className="mt-auto flex items-center justify-between pt-4 text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center space-x-2">
              <span>Long Text</span>
              <div className="h-2 w-2 rounded-full bg-gray-400 dark:bg-gray-500"></div>
            </div>
            <div className="flex items-center space-x-2">
              <span>{getCharacterCount()} / 5,000 characters</span>
              <div className="h-2 w-2 rounded-full bg-gray-200 dark:bg-gray-600"></div>
            </div>
          </div>
          <div className="mt-4 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="relative">
                <button className="flex items-center space-x-1 rounded-md border border-gray-200 dark:border-gray-700 px-2 py-1 text-xs hover:bg-gray-50 dark:hover:bg-gray-800">
                  <span>{language}</span>
                  <ChevronDown className="h-3 w-3" />
                </button>
              </div>
              <button
                className="rounded-md border border-gray-200 dark:border-gray-700 p-1 hover:bg-gray-50 dark:hover:bg-gray-800"
                onClick={handleDownload}
              >
                <Download className="h-4 w-4 text-gray-500 dark:text-gray-400" />
              </button>
            </div>
            <div className="flex items-center space-x-2">
              <button
                className={`rounded-md bg-sky-500 dark:bg-sky-600 px-3 py-1 text-sm text-white flex items-center hover:bg-sky-600 dark:hover:bg-sky-700 ${isGenerating ? "animate-pulse" : ""} disabled:opacity-50 disabled:cursor-not-allowed`}
                onClick={handleGenerate}
                disabled={isGenerateDisabled}
              >
                {isGenerating ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-1 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <RefreshCw className="h-4 w-4 mr-1" />
                    Generate
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Settings Panel */}
        <div className="w-80 overflow-auto p-4 bg-white dark:bg-gray-900">
          <div className="mb-4 flex space-x-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            <button
              className={`pb-2 text-sm ${activeTab === "settings" ? "border-b-2 border-black dark:border-white font-medium" : "text-gray-500 dark:text-gray-400"}`}
              onClick={() => setActiveTab("settings")}
            >
              Settings
            </button>
            <button
              className={`pb-2 text-sm ${activeTab === "history" ? "border-b-2 border-black dark:border-white font-medium" : "text-gray-500 dark:text-gray-400"}`}
              onClick={() => setActiveTab("history")}
            >
              History
            </button>
          </div>

          {activeTab === "settings" ? (
            <>
              {/* Model */}
              <div className="mb-6">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm">Model</span>
                  <div className="relative">
                    <select
                      className="flex w-40 appearance-none items-center justify-between rounded-md border border-gray-200 dark:border-gray-700 px-2 py-1 text-sm pr-8 bg-white dark:bg-gray-800"
                      value={baseModel}
                      onChange={(e) => handleModelChange(e.target.value)}
                    >
                      <option value="Marvis-AI/marvis-tts-100m-v0.2">Marvis-TTS-100m-v0.2</option>
                      <option value="Marvis-AI/marvis-tts-250m-v0.2">Marvis-TTS-250m-v0.2</option>
                      <option value="Marvis-AI/marvis-tts-250m-v0.1">Marvis-TTS-250m-v0.1</option>
                      <option value="mlx-community/Kokoro-82M-bf16">Kokoro</option>
                      <option value="mlx-community/Spark-TTS-0.5B-bf16">SparkTTS</option>
                      <option value="mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16">Qwen3-TTS VoiceDesign</option>
                    </select>
                    <ChevronDown className="absolute right-2 top-2 h-4 w-4 pointer-events-none" />
                  </div>
                </div>
              </div>

              {/* Quantization — Marvis only */}
              {isMarvisModel(baseModel) && (
                <div className="mb-6">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-sm">Quantization</span>
                    <div className="relative">
                      <select
                        className="flex w-40 appearance-none items-center justify-between rounded-md border border-gray-200 dark:border-gray-700 px-2 py-1 text-sm pr-8 bg-white dark:bg-gray-800"
                        value={quantization}
                        onChange={(e) => setQuantization(e.target.value)}
                      >
                        {getAvailableQuantizations(baseModel).map((quant) => (
                          <option key={quant} value={quant}>
                            {quant === "none" ? "None (bf16)" : quant.replace("bit", "-bit")}
                          </option>
                        ))}
                      </select>
                      <ChevronDown className="absolute right-2 top-2 h-4 w-4 pointer-events-none" />
                    </div>
                  </div>
                </div>
              )}

              {/* Selected model display */}
              <div className="mb-6">
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  <span className="font-medium">Selected Model:</span>{" "}
                  <span className="font-mono">{model}</span>
                </div>
              </div>

              {/* Voice Settings — model-specific */}
              <div className="mb-6 border-t border-gray-200 dark:border-gray-700 pt-4">
                <span className="block text-sm font-medium mb-3">Voice</span>

                {voiceSettings.family === "marvis" && (
                  <MarvisVoiceSettings
                    settings={voiceSettings}
                    onChange={(s) => setVoiceSettings({ family: "marvis", ...s })}
                  />
                )}
                {voiceSettings.family === "kokoro" && (
                  <KokoroVoiceSettings
                    settings={voiceSettings}
                    onChange={(s) => setVoiceSettings({ family: "kokoro", ...s })}
                  />
                )}
                {voiceSettings.family === "spark" && (
                  <SparkVoiceSettings
                    settings={voiceSettings}
                    onChange={(s) => setVoiceSettings({ family: "spark", ...s })}
                  />
                )}
                {voiceSettings.family === "qwen3" && (
                  <div className="space-y-3">
                    <div>
                      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                        Voice description
                      </label>
                      <textarea
                        className="w-full rounded-md border border-gray-200 dark:border-gray-700 p-2 text-sm bg-white dark:bg-gray-800 focus:border-blue-500 focus:outline-none"
                        value={voiceSettings.instruct}
                        onChange={(e) =>
                          setVoiceSettings({ ...voiceSettings, instruct: e.target.value })
                        }
                        placeholder="Describe the voice (e.g., 'A friendly, warm female voice')"
                        rows={3}
                      />
                      <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                        Use natural language: gender, age, tone, emotion, etc.
                      </p>
                    </div>

                    <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                      <label className="text-xs flex items-center gap-2 mb-1">
                        <input
                          type="checkbox"
                          checked={voiceSettings.useFixedSeed}
                          onChange={(e) =>
                            setVoiceSettings({ ...voiceSettings, useFixedSeed: e.target.checked })
                          }
                          className="rounded border-gray-300 text-sky-500 focus:ring-sky-500"
                        />
                        Consistent voice (same output each time)
                      </label>
                      <div className="flex items-center gap-2 mt-1">
                        <input
                          type="number"
                          value={voiceSettings.seed}
                          onChange={(e) =>
                            setVoiceSettings({ ...voiceSettings, seed: e.target.value })
                          }
                          placeholder="Optional: custom seed"
                          className="flex-1 rounded-md border border-gray-200 dark:border-gray-700 px-2 py-1 text-xs bg-white dark:bg-gray-800 focus:border-blue-500 focus:outline-none"
                          disabled={!voiceSettings.useFixedSeed}
                        />
                        <button
                          onClick={() =>
                            setVoiceSettings({
                              ...voiceSettings,
                              seed: Math.floor(Math.random() * 1000000).toString(),
                            })
                          }
                          disabled={!voiceSettings.useFixedSeed}
                          className="px-2 py-1 text-xs rounded-md border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50"
                        >
                          Random
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Speed */}
              <div className="mb-6">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm">Speed</span>
                  <div className="flex items-center">
                    <div className="flex space-x-2 mr-2">
                      {[0.5, 1, 1.5].map((s) => (
                        <button
                          key={s}
                          onClick={() => setSpeed(s)}
                          className={`px-2 py-0.5 text-xs rounded-md ${speed === s ? "bg-sky-100 dark:bg-sky-900 text-sky-600 dark:text-sky-300" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"}`}
                        >
                          {s}x
                        </button>
                      ))}
                    </div>
                    <span className="text-sm font-medium">{speed}x</span>
                  </div>
                </div>
                <div className="flex items-center">
                  <span className="text-xs text-gray-500 mr-2">Slow</span>
                  <RangeInput
                    min={0.5}
                    max={2}
                    step={0.1}
                    value={speed}
                    onChange={(e) => setSpeed(Number.parseFloat(e.target.value))}
                    ariaLabel="Speed control"
                  />
                  <span className="text-xs text-gray-500 ml-2">Fast</span>
                </div>
              </div>

              {/* Pitch — hidden for Spark and Qwen3 */}
              {showPitchSlider && (
                <div className="mb-6">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-sm">Pitch</span>
                    <div className="flex items-center">
                      <div className="flex space-x-2 mr-2">
                        {([-5, 0, 5] as const).map((p) => (
                          <button
                            key={p}
                            onClick={() => setPitch(p)}
                            className={`px-2 py-0.5 text-xs rounded-md ${pitch === p ? "bg-sky-100 dark:bg-sky-900 text-sky-600 dark:text-sky-300" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"}`}
                          >
                            {p === -5 ? "Low" : p === 0 ? "Normal" : "High"}
                          </button>
                        ))}
                      </div>
                      <span className="text-sm font-medium">{pitch}</span>
                    </div>
                  </div>
                  <div className="flex items-center">
                    <span className="text-xs text-gray-500 mr-2">-10</span>
                    <RangeInput
                      min={-10}
                      max={10}
                      step={1}
                      value={pitch}
                      onChange={(e) => setPitch(Number.parseInt(e.target.value))}
                      ariaLabel="Pitch control"
                    />
                    <span className="text-xs text-gray-500 ml-2">+10</span>
                  </div>
                </div>
              )}

              {/* Volume */}
              <div className="mb-4">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm">Volume</span>
                  <div className="flex items-center">
                    <div className="flex space-x-2 mr-2">
                      {([0.5, 1, 1.5] as const).map((v) => (
                        <button
                          key={v}
                          onClick={() => setVolume(v)}
                          className={`px-2 py-0.5 text-xs rounded-md ${volume === v ? "bg-sky-100 dark:bg-sky-900 text-sky-600 dark:text-sky-300" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"}`}
                        >
                          {v === 0.5 ? "Quiet" : v === 1 ? "Normal" : "Loud"}
                        </button>
                      ))}
                    </div>
                    <span className="text-sm font-medium">{volume}x</span>
                  </div>
                </div>
                <div className="flex items-center">
                  <span className="text-xs text-gray-500 mr-2">0</span>
                  <RangeInput
                    min={0}
                    max={2}
                    step={0.1}
                    value={volume}
                    onChange={(e) => setVolume(Number.parseFloat(e.target.value))}
                    ariaLabel="Volume control"
                  />
                  <span className="text-xs text-gray-500 ml-2">2</span>
                </div>
              </div>
            </>
          ) : (
            <div className="py-4 text-center text-sm text-gray-500 dark:text-gray-400">
              <p>No history available</p>
            </div>
          )}
        </div>
      </div>

      {/* Audio Player */}
      <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 w-full">
        <div className="flex items-center w-full px-0">
          <button
            className="flex h-14 w-14 items-center justify-center rounded-full bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 ml-4"
            onClick={handlePlayPause}
          >
            {isPlaying ? <Pause className="h-6 w-6" /> : <Play className="h-6 w-6" />}
          </button>

          <div className="flex flex-col justify-between h-full flex-1 px-4 py-2">
            <div className="flex items-center justify-between w-full">
              <div className="text-sm">
                {voiceDisplayLabel}: {text.length > 20 ? text.substring(0, 20) + "..." : text}
              </div>
              <div className="flex items-center space-x-2">
                <div className="text-xs text-gray-500 dark:text-gray-400 mr-2">How did this sound?</div>
                <button
                  className="rounded-md border border-gray-200 dark:border-gray-700 p-1 hover:bg-gray-50 dark:hover:bg-gray-800"
                  onClick={() => handleFeedback(true)}
                >
                  <ThumbsUp className={`h-4 w-4 ${liked === true ? "text-sky-500" : "text-gray-500 dark:text-gray-400"}`} />
                </button>
                <button
                  className="rounded-md border border-gray-200 dark:border-gray-700 p-1 hover:bg-gray-50 dark:hover:bg-gray-800"
                  onClick={() => handleFeedback(false)}
                >
                  <ThumbsDown className={`h-4 w-4 ${liked === false ? "text-sky-500" : "text-gray-500 dark:text-gray-400"}`} />
                </button>
                <button
                  className="rounded-md border border-gray-200 dark:border-gray-700 p-1 hover:bg-gray-50 dark:hover:bg-gray-800"
                  onClick={handleDownload}
                >
                  <Download className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                </button>
              </div>
            </div>

            <div className="flex items-center mt-2">
              <div
                className="flex-1 bg-gray-200 dark:bg-gray-700 h-1 rounded-full cursor-pointer relative"
                onClick={(e) => {
                  if (!audioRef.current || !audioRef.current.duration) return
                  const bar = e.currentTarget
                  const rect = bar.getBoundingClientRect()
                  const position = (e.clientX - rect.left) / rect.width
                  audioRef.current.currentTime = position * audioRef.current.duration
                  setCurrentTime(formatTime(Math.floor(audioRef.current.currentTime)))
                }}
              >
                <div
                  className="bg-black dark:bg-white h-1 rounded-full absolute top-0 left-0"
                  style={{
                    width: audioRef.current?.duration
                      ? `${(audioRef.current.currentTime / audioRef.current.duration) * 100}%`
                      : "0%",
                  }}
                ></div>
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400 ml-4 whitespace-nowrap mr-4">
                {currentTime} / {duration}
              </div>
            </div>
          </div>
        </div>
      </div>

      <audio ref={audioRef} className="hidden" />
    </LayoutWrapper>
  )
}
