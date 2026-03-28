# DeskMate AI 🤖 🔒

**Blazing Fast, 100% Private, On-Device Multimodal AI.**

DeskMate AI (VisionV) is a next-generation desktop assistant that runs entirely in your browser. Powered by the [RunAnywhere SDK](https://github.com/runanywhere/web), it leverages **WebGPU** and **Web Workers** to provide high-performance AI capabilities without ever sending your data to a server.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6?logo=typescript)
![Vite](https://img.shields.io/badge/Vite-6.0-646CFF?logo=vite)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local-green?logo=googlechrome)

---

## ✨ Key Features

- **🏠 100% On-Device**: All models run locally using WASM and WebGPU. Your data never leaves your machine.
- **👁️ Multimodal Vision (VLM)**: Upload images or use your camera to ask questions about what the AI "sees." (Powered by LFM2-VL).
- **🎙️ Real-time Voice**: Seamless STT (Speech-to-Text) and TTS (Text-to-Speech) for natural interactions. (Powered by Whisper & Piper).
- **🧠 Local Memory**: A persistent memory system that stores your notes and "memories" in `localStorage`.
- **⚡ WebGPU Accelerated**: Uses the full power of your hardware for low-latency, high-throughput inference.
- **✨ Explain Mode**: Simplified, beginner-friendly explanations of complex assistant responses.
- **🚀 Unified Interface**: A clean, modern chat experience with integrated vision and voice controls.

---

## 🛠️ Tech Stack

- **Framework**: [React 19](https://react.dev/)
- **Build Tool**: [Vite 6](https://vitejs.dev/)
- **Language**: [TypeScript](https://www.typescriptlang.org/)
- **AI Engine**: [@runanywhere/web](https://www.npmjs.com/package/@runanywhere/web)
- **Model Frameworks**: LlamaCPP (WASM) & ONNX Runtime (WASM)
- **Acceleration**: WebGPU / WebAssembly

---

## 🚀 Getting Started

### Prerequisites

- A modern browser with **WebGPU** support (Chrome 113+, Edge 113+ recommended).
- [Node.js](https://nodejs.org/) (v18 or higher)
- [npm](https://www.npmjs.com/) or [yarn](https://yarnpkg.com/)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/VisionV.git
    cd VisionV
    ```

2.  **Install dependencies**:
    ```bash
    npm install
    ```

3.  **Run the development server**:
    ```bash
    npm run dev
    ```

4.  **Open your browser**:
    Navigate to `http://localhost:5173`.

---

## 🤖 Models Used

| Modality | Model | Framework |
| :--- | :--- | :--- |
| **Language (LLM)** | Liquid AI LFM2 350M / 1.2B Tool | LlamaCPP |
| **Vision (VLM)** | Liquid AI LFM2-VL 450M | LlamaCPP |
| **Speech (STT)** | Whisper Tiny English | ONNX |
| **Speech (TTS)** | Piper TTS (Lessac) | ONNX |
| **VAD** | Silero VAD v5 | ONNX |

---

## 🔒 Privacy & Security

DeskMate AI is built with a **privacy-first** architecture. Unlike traditional AI assistants that rely on cloud APIs (like OpenAI or Anthropic), DeskMate AI:
- **Never uploads your audio, images, or text.**
- **Runs entirely offline** once the models are cached in your browser.
- **Requires no account or API keys.**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Built with ❤️ by Vishesh Aggarwal using the [RunAnywhere SDK](https://runanywhere.com).
