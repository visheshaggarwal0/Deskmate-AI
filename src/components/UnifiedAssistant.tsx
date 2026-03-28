import { useState, useRef, useEffect, useCallback } from 'react';
import { ModelCategory, VideoCapture, VoicePipeline, AudioCapture, AudioPlayback, SpeechActivity, ModelManager } from '@runanywhere/web';
import { TextGeneration, VLMWorkerBridge } from '@runanywhere/web-llamacpp';
import { VAD } from '@runanywhere/web-onnx';
import { useModelLoader } from '../hooks/useModelLoader';

type MessageType = 'user' | 'assistant' | 'system';
type MessageMode = 'text' | 'vision' | 'voice';

interface Message {
  id: string;
  type: MessageType;
  mode: MessageMode;
  text: string;
  imageData?: string;
  stats?: string;
  timestamp: Date;
}

type InputMode = 'text' | 'voice' | 'vision';
type VoiceState = 'idle' | 'loading-models' | 'listening' | 'processing' | 'speaking';

// Unified input types
interface CapturedFrame {
  rgbPixels: Uint8Array;
  width: number;
  height: number;
}

type UserInput =
  | { type: 'text'; data: { text: string } }
  | { type: 'voice'; data: { audioData: Float32Array } }
  | { type: 'vision'; data: { frame: CapturedFrame; prompt: string; imageUrl: string } };

export function UnifiedAssistant() {
  // Model loaders
  const llmLoader = useModelLoader(ModelCategory.Language, true);
  const vlmLoader = useModelLoader(ModelCategory.Multimodal);
  const sttLoader = useModelLoader(ModelCategory.SpeechRecognition, true);
  const ttsLoader = useModelLoader(ModelCategory.SpeechSynthesis, true);
  const vadLoader = useModelLoader(ModelCategory.Audio, true);

  // State
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [inputMode, setInputMode] = useState<InputMode>('text');
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Auto-focus input ref
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Explain Mode state
  const [lastAssistantMessage, setLastAssistantMessage] = useState<{ text: string; id: string } | null>(null);
  const [lastUserInput, setLastUserInput] = useState<string>('');
  
  // Cancel/interrupt state
  const cancelFunctionRef = useRef<(() => void) | null>(null);
  
  // Voice state
  const [voiceState, setVoiceState] = useState<VoiceState>('idle');
  const voicePipelineRef = useRef<VoicePipeline | null>(null);
  const audioCaptureRef = useRef<AudioCapture | null>(null);
  const vadUnsubRef = useRef<(() => void) | null>(null);
  
  // Vision state
  const [cameraActive, setCameraActive] = useState(false);
  const [visionPrompt, setVisionPrompt] = useState('What is in this image?');
  const videoMountRef = useRef<HTMLDivElement>(null);
  const captureRef = useRef<VideoCapture | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-focus input when switching back to text mode
  useEffect(() => {
    if (inputMode === 'text' && !isProcessing) {
      inputRef.current?.focus();
    }
  }, [inputMode, isProcessing]);

  // Cleanup
  useEffect(() => {
    return () => {
      const cam = captureRef.current;
      if (cam) {
        cam.stop();
        cam.videoElement.parentNode?.removeChild(cam.videoElement);
      }
      audioCaptureRef.current?.stop();
      vadUnsubRef.current?.();
    };
  }, []);

  // === HELPER: Add message ===
  const addMessage = useCallback((message: Omit<Message, 'id' | 'timestamp'>) => {
    const fullMessage: Message = {
      ...message,
      id: Date.now().toString() + Math.random(),
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, fullMessage]);
    return fullMessage.id;
  }, []);

  // === HELPER: Update message ===
  const updateMessage = useCallback((id: string, updates: Partial<Message>) => {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === id ? { ...msg, ...updates } : msg))
    );
  }, []);

  // === HELPER: Generate LLM response (with cancel support & batched updates) ===
  const generateLLMResponse = useCallback(async (
    prompt: string,
    mode: MessageMode,
    isExplainMode: boolean = false
  ): Promise<{ text: string; stats: string }> => {
    // Ensure LLM is loaded
    if (llmLoader.state !== 'ready') {
      const ok = await llmLoader.ensure();
      if (!ok) throw new Error('Failed to load LLM');
    }

    // Create placeholder assistant message
    const assistantId = addMessage({
      type: 'assistant',
      mode,
      text: '',
    });

    const t0 = performance.now();
    const systemPrefix = isExplainMode
      ? 'Explain in very simple terms with bullet points and examples. Be clear and beginner-friendly. '
      : 'Be concise. Use bullet points and short paragraphs. No filler. ';

    const fullPrompt = systemPrefix + prompt;

    const { stream, result: resultPromise, cancel } = await TextGeneration.generateStream(fullPrompt, {
      maxTokens: isExplainMode ? 400 : 256,
      temperature: isExplainMode ? 0.5 : 0.4,
    });

    // Store cancel function so UI can interrupt
    cancelFunctionRef.current = cancel;

    // Batched UI updates via requestAnimationFrame for smoother streaming
    let accumulated = '';
    let wasCancelled = false;
    let frameId: number | null = null;
    let pendingText = '';

    const flushUpdate = () => {
      if (pendingText) {
        updateMessage(assistantId, { text: pendingText });
        frameId = null;
      }
    };

    try {
      for await (const token of stream) {
        accumulated += token;
        pendingText = accumulated;
        if (!frameId) {
          frameId = requestAnimationFrame(flushUpdate);
        }
      }
      // Final flush
      if (frameId) cancelAnimationFrame(frameId);
      updateMessage(assistantId, { text: accumulated });
    } catch (err) {
      if (frameId) cancelAnimationFrame(frameId);
      if ((err as Error).message?.includes('cancel')) {
        wasCancelled = true;
      } else {
        throw err;
      }
    }

    // Get final result
    try {
      const result = await resultPromise;
      const tokensPerSec = result.tokensPerSecond?.toFixed(1) || '0';
      const latency = ((performance.now() - t0) / 1000).toFixed(2);
      const stats = wasCancelled
        ? `Cancelled · ${latency}s · ${tokensPerSec} tok/s`
        : `${latency}s · ${tokensPerSec} tok/s`;

      const finalText = wasCancelled
        ? accumulated + '\n\n*[Response stopped]*'
        : (result.text || accumulated);

      updateMessage(assistantId, { text: finalText, stats });

      // Track last assistant message (only for non-explain messages)
      if (!isExplainMode && mode === 'text' && !wasCancelled) {
        setLastAssistantMessage({ text: finalText, id: assistantId });
      }

      return { text: finalText, stats };
    } catch (err) {
      if ((err as Error).message?.includes('cancel')) {
        wasCancelled = true;
        const finalText = accumulated + '\n\n*[Response stopped]*';
        const latency = ((performance.now() - t0) / 1000).toFixed(2);
        updateMessage(assistantId, { text: finalText, stats: `Cancelled · ${latency}s` });
        return { text: finalText, stats: `Cancelled · ${latency}s` };
      }
      throw err;
    } finally {
      cancelFunctionRef.current = null;
    }
  }, [llmLoader, addMessage, updateMessage]);

  // === UNIFIED INPUT HANDLER ===
  const handleUserInput = useCallback(async (input: UserInput) => {
    if (isProcessing) return;
    setIsProcessing(true);

    try {
      switch (input.type) {
        case 'text': {
          // Add user message
          addMessage({
            type: 'user',
            mode: 'text',
            text: input.data.text,
          });

          // Track last user input for memory feature
          setLastUserInput(input.data.text);

          // Generate LLM response directly
          await generateLLMResponse(input.data.text, 'text');
          break;
        }

        case 'vision': {
          // Add user message with image
          addMessage({
            type: 'user',
            mode: 'vision',
            text: input.data.prompt,
            imageData: input.data.imageUrl,
          });

          // Ensure VLM is loaded
          if (vlmLoader.state !== 'ready') {
            const ok = await vlmLoader.ensure();
            if (!ok) throw new Error('Failed to load VLM');
          }

          // Process vision through VLM
          const t0 = performance.now();
          const bridge = VLMWorkerBridge.shared;
          if (!bridge.isModelLoaded) {
            throw new Error('VLM model not loaded in worker');
          }

          const res = await bridge.process(
            input.data.frame.rgbPixels,
            input.data.frame.width,
            input.data.frame.height,
            input.data.prompt,
            { maxTokens: 256, temperature: 0.6 },
          );

          const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
          const stats = `${elapsed}s`;

          // Add vision response
          addMessage({
            type: 'assistant',
            mode: 'vision',
            text: res.text || '(no response)',
            stats,
          });
          break;
        }

        case 'voice': {
          // Ensure all voice models are loaded
          const anyMissing = !ModelManager.getLoadedModel(ModelCategory.Audio)
            || !ModelManager.getLoadedModel(ModelCategory.SpeechRecognition)
            || !ModelManager.getLoadedModel(ModelCategory.Language)
            || !ModelManager.getLoadedModel(ModelCategory.SpeechSynthesis);

          if (anyMissing) {
            setVoiceState('loading-models');
            const results = await Promise.all([
              vadLoader.ensure(),
              sttLoader.ensure(),
              llmLoader.ensure(),
              ttsLoader.ensure(),
            ]);

            if (!results.every(Boolean)) {
              throw new Error('Failed to load voice models');
            }
          }

          setVoiceState('processing');

          // Initialize pipeline if needed
          if (!voicePipelineRef.current) {
            voicePipelineRef.current = new VoicePipeline();
          }

          const pipeline = voicePipelineRef.current;
          let transcriptText = '';
          let responseText = '';

          // Process through voice pipeline (STT → LLM → TTS)
          await pipeline.processTurn(input.data.audioData, {
            maxTokens: 60,
            temperature: 0.7,
            systemPrompt: 'You are a helpful desktop AI assistant. Keep responses concise — 1-2 sentences max.',
          }, {
            onTranscription: (text) => {
              transcriptText = text;
              // Add user message with transcript
              addMessage({
                type: 'user',
                mode: 'voice',
                text: `🎙️ ${text}`,
              });
            },
            onResponseToken: (_token, accumulated) => {
              responseText = accumulated;
            },
            onResponseComplete: (text) => {
              responseText = text;
            },
            onSynthesisComplete: async (audio, sampleRate) => {
              // Speak the response
              setVoiceState('speaking');
              const player = new AudioPlayback({ sampleRate });
              await player.play(audio, sampleRate);
              player.dispose();

              // Add assistant message
              addMessage({
                type: 'assistant',
                mode: 'voice',
                text: responseText,
              });

              setVoiceState('idle');
            },
            onStateChange: (_state) => {
              // Could add more detailed state updates
            },
          });
          break;
        }
      }
    } catch (err) {
      // Add error message
      addMessage({
        type: 'system',
        mode: input.type,
        text: `Error: ${err instanceof Error ? err.message : String(err)}`,
      });

      if (input.type === 'voice') {
        setVoiceState('idle');
      }
    } finally {
      if (input.type !== 'voice') {
        setIsProcessing(false);
      }
    }
  }, [isProcessing, llmLoader, vlmLoader, sttLoader, ttsLoader, vadLoader, addMessage, generateLLMResponse]);

  // === TEXT INPUT HANDLER ===
  const sendTextMessage = useCallback(async () => {
    if (!inputText.trim() || isProcessing) return;

    const text = inputText;
    setInputText('');

    await handleUserInput({
      type: 'text',
      data: { text },
    });
  }, [inputText, isProcessing, handleUserInput]);

  // === CANCEL / INTERRUPT HANDLER ===
  const handleCancel = useCallback(() => {
    if (cancelFunctionRef.current) {
      cancelFunctionRef.current();
      cancelFunctionRef.current = null;
    }
    setIsProcessing(false);
  }, []);

  // === MEMORY SYSTEM (localStorage) ===
  const MEMORY_KEY = 'deskmate_memory';

  interface MemoryNote {
    id: string;
    text: string;
    createdAt: string;
    mode: 'text' | 'vision' | 'voice';
  }

  const [memoryNotes, setMemoryNotes] = useState<MemoryNote[]>(() => {
    try {
      const saved = localStorage.getItem(MEMORY_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const saveMemory = useCallback((text: string, mode: 'text' | 'vision' | 'voice' = 'text') => {
    if (!text.trim()) {
      addMessage({
        type: 'system',
        mode: 'text',
        text: 'Nothing to save.',
      });
      return;
    }

    const note: MemoryNote = {
      id: Date.now().toString(),
      text: text.trim(),
      createdAt: new Date().toLocaleString(),
      mode,
    };

    const updated = [note, ...memoryNotes];
    setMemoryNotes(updated);
    localStorage.setItem(MEMORY_KEY, JSON.stringify(updated));

    addMessage({
      type: 'system',
      mode: 'text',
      text: `🧠 Saved to memory (${updated.length} notes total)`,
    });
  }, [memoryNotes, addMessage]);

  const recallMemory = useCallback(() => {
    if (memoryNotes.length === 0) {
      addMessage({
        type: 'system',
        mode: 'text',
        text: '📂 No saved memories yet. Use "Remember this" to save notes.',
      });
      return;
    }

    let display = `📂 **Your Saved Memories** (${memoryNotes.length} notes)\n\n`;
    memoryNotes.forEach((note, i) => {
      const modeIcon = note.mode === 'voice' ? '🎙️' : note.mode === 'vision' ? '📷' : '💬';
      display += `${i + 1}. ${modeIcon} ${note.text}\n   Saved: ${note.createdAt}\n\n`;
    });

    addMessage({
      type: 'assistant',
      mode: 'text',
      text: display,
    });
  }, [memoryNotes, addMessage]);

  const deleteMemory = useCallback((id: string) => {
    const updated = memoryNotes.filter((n) => n.id !== id);
    setMemoryNotes(updated);
    localStorage.setItem(MEMORY_KEY, JSON.stringify(updated));

    addMessage({
      type: 'system',
      mode: 'text',
      text: `🗑️ Memory deleted (${updated.length} notes remaining)`,
    });
  }, [memoryNotes, addMessage]);

  const clearAllMemories = useCallback(() => {
    setMemoryNotes([]);
    localStorage.removeItem(MEMORY_KEY);

    addMessage({
      type: 'system',
      mode: 'text',
      text: '🗑️ All memories cleared.',
    });
  }, [addMessage]);

  // === EXPLAIN MODE HANDLER ===
  const handleExplain = useCallback(async () => {
    if (!lastAssistantMessage || isProcessing) return;

    setIsProcessing(true);

    try {
      // Add system message indicating explain mode
      addMessage({
        type: 'system',
        mode: 'text',
        text: '✨ Generating simplified explanation...',
      });

      // Craft the explain prompt
      const explainPrompt = `Explain the following response in very simple terms, as if explaining to a beginner. Break it down with clear bullet points and examples where helpful:

${lastAssistantMessage.text}

Please provide:
1. A simple summary (1-2 sentences)
2. Key points as bullet points
3. Any helpful examples or analogies

Keep it clear, concise, and beginner-friendly.`;

      // Generate explanation using LLM
      await generateLLMResponse(explainPrompt, 'text', true);
    } catch (err) {
      addMessage({
        type: 'system',
        mode: 'text',
        text: `Explain error: ${err instanceof Error ? err.message : String(err)}`,
      });
    } finally {
      setIsProcessing(false);
    }
  }, [lastAssistantMessage, isProcessing, addMessage, generateLLMResponse]);

  // === VISION HANDLERS ===
  const startCamera = useCallback(async () => {
    try {
      const vc = new VideoCapture({ facingMode: 'environment' });
      await vc.start();
      captureRef.current = vc;

      const mount = videoMountRef.current;
      if (mount) {
        const el = vc.videoElement;
        el.style.width = '100%';
        el.style.borderRadius = '8px';
        
        // Ensure video attributes for proper playback
        el.setAttribute('autoplay', 'true');
        el.setAttribute('playsinline', 'true');
        el.setAttribute('muted', 'true');
        
        mount.appendChild(el);

        // Wait for video to be ready
        await new Promise<void>((resolve) => {
          if (el.readyState >= 2) {
            resolve();
          } else {
            el.addEventListener('loadeddata', () => resolve(), { once: true });
          }
        });
      }

      setCameraActive(true);

      addMessage({
        type: 'system',
        mode: 'vision',
        text: 'Camera started. Click "Capture & Analyze" or upload an image.',
      });
    } catch (err) {
      addMessage({
        type: 'system',
        mode: 'vision',
        text: `Camera not working. Please upload an image instead. Error: ${err instanceof Error ? err.message : String(err)}`,
      });
      setCameraActive(false);
    }
  }, [addMessage]);

  const stopCamera = useCallback(() => {
    const cam = captureRef.current;
    if (cam) {
      cam.stop();
      cam.videoElement.parentNode?.removeChild(cam.videoElement);
      captureRef.current = null;
    }
    setCameraActive(false);
  }, []);

  const captureAndAnalyze = useCallback(async () => {
    const cam = captureRef.current;
    if (!cam?.isCapturing || isProcessing) return;

    try {
      const frame = cam.captureFrame(256);
      if (!frame) throw new Error('Failed to capture frame');

      // Create canvas for preview image
      const canvas = document.createElement('canvas');
      canvas.width = frame.width;
      canvas.height = frame.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Canvas context failed');

      const imageData = ctx.createImageData(frame.width, frame.height);
      for (let i = 0; i < frame.rgbPixels.length; i++) {
        imageData.data[i * 4 + 0] = frame.rgbPixels[i * 3 + 0];
        imageData.data[i * 4 + 1] = frame.rgbPixels[i * 3 + 1];
        imageData.data[i * 4 + 2] = frame.rgbPixels[i * 3 + 2];
        imageData.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

      await handleUserInput({
        type: 'vision',
        data: {
          frame: {
            rgbPixels: frame.rgbPixels,
            width: frame.width,
            height: frame.height,
          },
          prompt: visionPrompt,
          imageUrl: dataUrl,
        },
      });
    } catch (err) {
      addMessage({
        type: 'system',
        mode: 'vision',
        text: `Vision error: ${err instanceof Error ? err.message : String(err)}`,
      });
    }
  }, [isProcessing, visionPrompt, handleUserInput, addMessage]);

  // === FILE UPLOAD HANDLER ===
  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || isProcessing) return;

    // Check if it's an image
    if (!file.type.startsWith('image/')) {
      addMessage({
        type: 'system',
        mode: 'vision',
        text: 'Please select an image file (JPG, PNG, etc.)',
      });
      return;
    }

    try {
      // Read file as data URL for preview
      const reader = new FileReader();
      reader.onload = async (e) => {
        const dataUrl = e.target?.result as string;

        // Create image element to load the file
        const img = new Image();
        img.onload = async () => {
          // Create canvas to extract RGB data
          const canvas = document.createElement('canvas');
          const targetSize = 256;
          
          // Calculate dimensions maintaining aspect ratio
          let width = img.width;
          let height = img.height;
          if (width > height) {
            if (width > targetSize) {
              height = (height * targetSize) / width;
              width = targetSize;
            }
          } else {
            if (height > targetSize) {
              width = (width * targetSize) / height;
              height = targetSize;
            }
          }

          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          if (!ctx) throw new Error('Canvas context failed');

          // Draw image
          ctx.drawImage(img, 0, 0, width, height);

          // Extract RGB pixel data
          const imageData = ctx.getImageData(0, 0, width, height);
          const rgbPixels = new Uint8Array(width * height * 3);
          
          for (let i = 0; i < imageData.data.length / 4; i++) {
            rgbPixels[i * 3 + 0] = imageData.data[i * 4 + 0]; // R
            rgbPixels[i * 3 + 1] = imageData.data[i * 4 + 1]; // G
            rgbPixels[i * 3 + 2] = imageData.data[i * 4 + 2]; // B
          }

          // Process through unified input
          await handleUserInput({
            type: 'vision',
            data: {
              frame: {
                rgbPixels,
                width,
                height,
              },
              prompt: visionPrompt,
              imageUrl: dataUrl,
            },
          });
        };

        img.onerror = () => {
          addMessage({
            type: 'system',
            mode: 'vision',
            text: 'Failed to load image file',
          });
        };

        img.src = dataUrl;
      };

      reader.onerror = () => {
        addMessage({
          type: 'system',
          mode: 'vision',
          text: 'Failed to read image file',
        });
      };

      reader.readAsDataURL(file);
    } catch (err) {
      addMessage({
        type: 'system',
        mode: 'vision',
        text: `Upload error: ${err instanceof Error ? err.message : String(err)}`,
      });
    }

    // Reset file input
    event.target.value = '';
  }, [isProcessing, visionPrompt, handleUserInput, addMessage]);

  // === VOICE HANDLERS ===
  const startVoiceInteraction = useCallback(async () => {
    if (voiceState !== 'idle') return;

    setVoiceState('listening');

    addMessage({
      type: 'system',
      mode: 'voice',
      text: '🎙️ Listening... Speak now.',
    });

    const mic = new AudioCapture({ sampleRate: 16000 });
    audioCaptureRef.current = mic;

    VAD.reset();

    vadUnsubRef.current = VAD.onSpeechActivity((activity) => {
      if (activity === SpeechActivity.Ended) {
        const segment = VAD.popSpeechSegment();
        if (segment && segment.samples.length > 1600) {
          // Stop listening and process speech
          mic.stop();
          vadUnsubRef.current?.();

          handleUserInput({
            type: 'voice',
            data: { audioData: segment.samples },
          });
        }
      }
    });

    await mic.start(
      (chunk) => { VAD.processSamples(chunk); },
      (_level) => { /* Could visualize audio level */ },
    );
  }, [voiceState, handleUserInput, addMessage]);

  // === QUICK ACTIONS ===
  const quickActions = [
    { label: 'Explain recursion', prompt: 'Explain recursion in simple terms with examples.' },
    { label: 'How does WebGPU work?', prompt: 'How does WebGPU work and why is it fast?' },
    { label: 'Tell me a fun fact', prompt: 'Tell me a surprising fun fact about space.' },
    { label: 'Write a haiku', prompt: 'Write a haiku about artificial intelligence.' },
  ];

  const handleQuickAction = useCallback((prompt: string) => {
    setInputText(prompt);
    setInputMode('text');
    handleUserInput({ type: 'text', data: { text: prompt } });
  }, [handleUserInput]);
  const showTextBanner = inputMode === 'text' && llmLoader.state !== 'ready' && llmLoader.state !== 'idle';
  const showVisionBanner = inputMode === 'vision' && vlmLoader.state !== 'ready' && vlmLoader.state !== 'idle';
  const showVoiceBanner = inputMode === 'voice' && (
    llmLoader.state !== 'ready' || sttLoader.state !== 'ready' || 
    ttsLoader.state !== 'ready' || vadLoader.state !== 'ready'
  ) && voiceState === 'loading-models';

  return (
    <div className="unified-assistant">
      {/* Model loading banners */}
      {showTextBanner && (
        <div className="model-banner">
          <span>
            {llmLoader.state === 'downloading' && `Downloading LLM... ${(llmLoader.progress * 100).toFixed(0)}%`}
            {llmLoader.state === 'loading' && 'Loading LLM into engine...'}
            {llmLoader.state === 'error' && `Error: ${llmLoader.error}`}
          </span>
          {llmLoader.state === 'downloading' && (
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${llmLoader.progress * 100}%` }} />
            </div>
          )}
        </div>
      )}

      {showVisionBanner && (
        <div className="model-banner">
          <span>
            {vlmLoader.state === 'downloading' && `Downloading VLM... ${(vlmLoader.progress * 100).toFixed(0)}%`}
            {vlmLoader.state === 'loading' && 'Loading VLM into engine...'}
            {vlmLoader.state === 'error' && `Error: ${vlmLoader.error}`}
          </span>
          {vlmLoader.state === 'downloading' && (
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${vlmLoader.progress * 100}%` }} />
            </div>
          )}
        </div>
      )}

      {showVoiceBanner && (
        <div className="model-banner">
          <span>Loading voice models...</span>
        </div>
      )}

      {/* Response Panel */}
      <div className="response-panel">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-logo">🤖</div>
            <h3>DeskMate AI</h3>
            <p>Your offline AI desktop companion</p>
            <div className="quick-actions-grid">
              {quickActions.map((action, i) => (
                <button
                  key={i}
                  className="quick-action-btn"
                  onClick={() => handleQuickAction(action.prompt)}
                >
                  {action.label}
                </button>
              ))}
            </div>
            <div className="empty-features">
              <span>💬 Chat</span>
              <span>🎙️ Voice</span>
              <span>📷 Vision</span>
              <span>🧠 Memory</span>
            </div>
          </div>
        ) : (
          <div className="message-list">
            {messages.map((msg, index) => (
              <div key={msg.id}>
                <div className={`message message-${msg.type}`}>
                  <div className="message-bubble">
                    {msg.imageData && (
                      <img
                        src={msg.imageData}
                        alt="Captured frame"
                        className="message-image"
                      />
                    )}
                    <div className="message-text">{msg.text}</div>
                    {msg.stats && (
                      <div className="message-stats">{msg.stats}</div>
                    )}
                  </div>
                </div>
                {/* Show Explain button after assistant text messages */}
                {msg.type === 'assistant' && 
                 msg.mode === 'text' && 
                 msg.id === lastAssistantMessage?.id && 
                 !isProcessing && (
                  <div className="explain-button-container">
                    <button 
                      className="btn-explain"
                      onClick={handleExplain}
                      title="Get a simplified explanation"
                    >
                      ✨ Explain
                    </button>
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Vision camera preview (shown when vision mode is active and camera is on) */}
      {inputMode === 'vision' && cameraActive && (
        <div className="vision-preview">
          <div className="vision-camera" ref={videoMountRef} />
          <input
            type="text"
            className="vision-prompt"
            value={visionPrompt}
            onChange={(e) => setVisionPrompt(e.target.value)}
            placeholder="Enter vision prompt..."
          />
        </div>
      )}

      {/* Input Controls */}
      <div className="input-controls">
        <div className="mode-selector">
          <button
            className={`mode-btn ${inputMode === 'text' ? 'active' : ''}`}
            onClick={() => setInputMode('text')}
            title="Text chat"
          >
            💬
          </button>
          <button
            className={`mode-btn ${inputMode === 'voice' ? 'active' : ''}`}
            onClick={() => setInputMode('voice')}
            title="Voice interaction"
          >
            🎙️
          </button>
          <button
            className={`mode-btn ${inputMode === 'vision' ? 'active' : ''}`}
            onClick={() => setInputMode('vision')}
            title="Vision analysis"
          >
            📷
          </button>
        </div>

        {/* Memory & Action buttons */}
        <div className="action-buttons">
          <button
            className="btn btn-sm btn-memory"
            onClick={() => {
              const textToSave = lastUserInput || inputText;
              if (!textToSave.trim()) {
                addMessage({
                  type: 'system',
                  mode: 'text',
                  text: 'Type something first to save.',
                });
                return;
              }
              saveMemory(textToSave, 'text');
            }}
            disabled={isProcessing}
            title="Save last input to memory"
          >
            🧠 Remember
          </button>
          <button
            className="btn btn-sm btn-memory"
            onClick={recallMemory}
            disabled={isProcessing}
            title="Recall saved memories"
          >
            📂 Recall ({memoryNotes.length})
          </button>
          {memoryNotes.length > 0 && (
            <button
              className="btn btn-sm btn-memory btn-memory-clear"
              onClick={clearAllMemories}
              disabled={isProcessing}
              title="Clear all memories"
            >
              🗑️
            </button>
          )}
        </div>

        {inputMode === 'text' && (
          <div className="text-input-area">
            <input
              ref={inputRef}
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !isProcessing && sendTextMessage()}
              placeholder="Ask anything..."
              disabled={isProcessing}
            />
            {isProcessing ? (
              <button
                className="btn btn-cancel"
                onClick={handleCancel}
              >
                ✕ Stop
              </button>
            ) : (
              <button
                className="btn btn-primary"
                onClick={sendTextMessage}
                disabled={!inputText.trim()}
              >
                Send
              </button>
            )}
          </div>
        )}

        {inputMode === 'voice' && (
          <div className="voice-input-area">
            {voiceState !== 'idle' && voiceState !== 'loading-models' ? (
              <button
                className="btn btn-cancel"
                onClick={() => {
                  audioCaptureRef.current?.stop();
                  vadUnsubRef.current?.();
                  setVoiceState('idle');
                  setIsProcessing(false);
                  addMessage({
                    type: 'system',
                    mode: 'voice',
                    text: 'Voice interaction stopped.',
                  });
                }}
              >
                ✕ Stop Voice
              </button>
            ) : (
              <button
                className="btn btn-primary btn-lg"
                onClick={startVoiceInteraction}
                disabled={voiceState !== 'idle'}
              >
                {voiceState === 'idle' && '🎙️ Start Voice'}
                {voiceState === 'loading-models' && 'Loading Models...'}
              </button>
            )}
          </div>
        )}

        {inputMode === 'vision' && (
          <div className="vision-input-area">
            {/* File upload - always visible */}
            <label htmlFor="image-upload">
              <input
                id="image-upload"
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="file-input-hidden"
                disabled={isProcessing}
              />
              <button
                className="btn btn-primary"
                onClick={() => document.getElementById('image-upload')?.click()}
                disabled={isProcessing}
                type="button"
              >
                📂 Upload Image
              </button>
            </label>

            {/* Camera controls */}
            {!cameraActive ? (
              <button
                className="btn"
                onClick={startCamera}
                disabled={isProcessing}
              >
                📷 Use Camera
              </button>
            ) : (
              <>
                <button
                  className={isProcessing ? 'btn btn-cancel' : 'btn btn-primary'}
                  onClick={isProcessing ? handleCancel : captureAndAnalyze}
                >
                  {isProcessing ? '✕ Stop' : '✨ Capture & Analyze'}
                </button>
                <button
                  className="btn btn-sm"
                  onClick={stopCamera}
                  disabled={isProcessing}
                >
                  Stop
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
