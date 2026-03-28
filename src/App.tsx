import { useState, useEffect } from 'react';
import { initSDK, getAccelerationMode } from './runanywhere';
import { UnifiedAssistant } from './components/UnifiedAssistant';
import { ErrorBoundary } from './components/ErrorBoundary';

export function App() {
  const [sdkReady, setSdkReady] = useState(false);
  const [sdkError, setSdkError] = useState<string | null>(null);

  useEffect(() => {
    initSDK()
      .then(() => setSdkReady(true))
      .catch((err) => setSdkError(err instanceof Error ? err.message : String(err)));
  }, []);

  if (sdkError) {
    return (
      <div className="app-loading">
        <h2>SDK Error</h2>
        <p className="error-text">{sdkError}</p>
      </div>
    );
  }

  if (!sdkReady) {
    return (
      <div className="app-loading">
        <div className="spinner" />
        <h2>Loading RunAnywhere SDK...</h2>
        <p>Initializing on-device AI engine</p>
      </div>
    );
  }

  const accel = getAccelerationMode();

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-left">
          <h1>DeskMate AI 🔒</h1>
          <div className="privacy-badge">
            <span>🔒 Runs 100% locally</span>
          </div>
        </div>
        {accel && <span className="badge">{accel === 'webgpu' ? 'WebGPU' : 'CPU'}</span>}
      </header>

      <main className="tab-content">
        <ErrorBoundary>
          <UnifiedAssistant />
        </ErrorBoundary>
      </main>
    </div>
  );
}
