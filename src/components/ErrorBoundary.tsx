import React from 'react';

interface State {
  hasError: boolean;
  error: string | null;
}

export class ErrorBoundary extends React.Component<React.PropsWithChildren, State> {
  constructor(props: React.PropsWithChildren) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error: error.message };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('DeskMate AI Error:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <div className="error-boundary-content">
            <span className="error-icon">⚠️</span>
            <h2>Something went wrong</h2>
            <p className="error-text">{this.state.error}</p>
            <button className="btn btn-primary" onClick={this.handleReset}>
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
