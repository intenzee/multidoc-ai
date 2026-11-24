import { Component } from "react";

// If a component crashes, show a short message instead of a blank page.
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { failed: false };
  }

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error) {
    console.error(error); // details stay in the browser console
  }

  render() {
    if (this.state.failed) {
      return (
        <div className="card" role="alert" style={{ margin: 16 }}>
          <strong>Something went wrong in this part of the page.</strong>
          <p className="muted">Your documents are safe. Reload the page to continue.</p>
          <button onClick={() => window.location.reload()}>Reload</button>
        </div>
      );
    }
    return this.props.children;
  }
}
