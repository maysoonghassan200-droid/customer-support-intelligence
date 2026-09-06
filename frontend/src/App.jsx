import { useState } from "react";
import "./App.css";

function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Hello! I'm your AI support assistant. How can I help you today?",
    },
  ]);
  const [queue, setQueue] = useState("-");
  const [priority, setPriority] = useState("-");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!message.trim() || loading) return;

    const userMessage = message;

    setMessages((prev) => [
      ...prev,
      { role: "user", text: userMessage },
    ]);

    setMessage("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8010/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: "dashboard_user",
          message: userMessage,
        }),
      });

      if (!response.ok) {
        throw new Error("API request failed");
      }

      const data = await response.json();

      setQueue(data.queue);
      setPriority(data.priority);

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.answer },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Unable to connect to the support API.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <div>
          <h1>Customer Support Intelligence</h1>
          <p>AI-powered ticket triage & RAG assistant</p>
        </div>
        <span className="status">● API Connected</span>
      </header>

      <main>
        <section className="dashboard">
          <h2>Ticket Intelligence</h2>

          <div className="cards">
            <div className="card">
              <span>Predicted Queue</span>
              <strong>{queue}</strong>
            </div>

            <div className="card">
              <span>Priority</span>
              <strong>{priority}</strong>
            </div>

            <div className="card">
              <span>Knowledge Base</span>
              <strong>8,320 Chunks</strong>
            </div>
          </div>

          <div className="info">
            <h3>System Pipeline</h3>
            <p>
              Ticket → ML Triage → FAISS Retrieval → Gemini →
              Support Response
            </p>
          </div>
        </section>

        <section className="chat">
          <div className="chat-header">
            <div>
              <h2>Support Assistant</h2>
              <span>RAG + Conversation Memory</span>
            </div>
          </div>

          <div className="messages">
            {messages.map((item, index) => (
              <div
                key={index}
                className={`message ${item.role}`}
              >
                {item.text}
              </div>
            ))}

            {loading && (
              <div className="message assistant">
                Thinking...
              </div>
            )}
          </div>

          <div className="input-area">
            <input
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") sendMessage();
              }}
              placeholder="Describe your support issue..."
            />

            <button onClick={sendMessage} disabled={loading}>
              Send
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;