
import { useEffect, useRef, useState } from 'react'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000'

function App() {
  const [activeTab, setActiveTab] = useState('Dashboard')
  const [stats, setStats] = useState({ total_documents: 0, total_chunks: 0 })
  const [documents, setDocuments] = useState([])
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I am CyberShield AI. Upload your cybersecurity documents and ask me questions about vulnerabilities, OWASP risks, security controls, and remediation.',
    },
  ])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const fileInput = useRef(null)
  const chatEnd = useRef(null)

  const loadDashboard = async () => {
    try {
      const [statsResponse, docsResponse] = await Promise.all([
        fetch(`${API_BASE}/api/documents/stats`),
        fetch(`${API_BASE}/api/documents/`),
      ])

      if (!statsResponse.ok || !docsResponse.ok) {
        throw new Error('Could not load backend data.')
      }

      const statsData = await statsResponse.json()
      const docsData = await docsResponse.json()

      setStats(statsData)
      setDocuments(docsData.documents || [])
      setError('')
    } catch (err) {
      setError('Backend connection failed. Make sure FastAPI is running.')
    }
  }

  useEffect(() => {
    loadDashboard()
  }, [])

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleUpload = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return

    setUploading(true)
    setError('')
    setSuccess('')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_BASE}/api/documents/upload`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Document upload failed.')
      }

      setSuccess(`${data.filename} uploaded and indexed successfully.`)
      await loadDashboard()
    } catch (err) {
      setError(err.message || 'Upload failed.')
    } finally {
      setUploading(false)
      event.target.value = ''
    }
  }

  const handleDelete = async (documentId) => {
    const confirmed = window.confirm(
      'Are you sure you want to delete this document and its indexed chunks?'
    )

    if (!confirmed) return

    setError('')
    setSuccess('')

    try {
      const response = await fetch(
        `${API_BASE}/api/documents/${encodeURIComponent(documentId)}`,
        { method: 'DELETE' }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Could not delete document.')
      }

      setSuccess('Document deleted successfully.')
      await loadDashboard()
    } catch (err) {
      setError(err.message)
    }
  }

  const handleAsk = async (event) => {
    event.preventDefault()

    const userQuestion = question.trim()
    if (!userQuestion || loading) return

    setMessages((previous) => [
      ...previous,
      { role: 'user', content: userQuestion },
    ])
    setQuestion('')
    setLoading(true)
    setError('')

    try {
      const response = await fetch(`${API_BASE}/api/chat/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userQuestion,
          top_k: 5,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'AI request failed.')
      }

     
const result = data.data || data;

setMessages((previous) => [
  ...previous,
  {
    role: 'assistant',
    content: result.answer || 'No answer was returned.',
    sources: result.sources || [],
  },
]);
    } catch (err) {
      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          content: `I couldn't process your question: ${err.message}`,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const navigation = [
    { name: 'Dashboard', icon: '◫' },
    { name: 'Documents', icon: '▤' },
    { name: 'AI Assistant', icon: '✧' },
  ]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">⌘</div>
          <div>
            <h2>CyberShield<span> AI</span></h2>
            <p>INTELLIGENCE PLATFORM</p>
          </div>
        </div>

        <div className="side-label">WORKSPACE</div>

        <nav className="navigation">
          {navigation.map((item) => (
            <button
              key={item.name}
              className={`nav-item ${activeTab === item.name ? 'active' : ''}`}
              onClick={() => setActiveTab(item.name)}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.name}
              {item.name === 'Documents' && (
                <span className="nav-count">{documents.length}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="security-card">
            <div className="security-icon">✦</div>
            <h3>AI Security Engine</h3>
            <p>Powered by retrieval-augmented generation.</p>
            <span className="online-indicator">● System ready</span>
          </div>
          <div className="user-profile">
            <div className="avatar">CS</div>
            <div>
              <strong>Security Analyst</strong>
              <span>Local workspace</span>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="breadcrumb">Workspace <span>/</span> {activeTab}</div>
            <h1>{activeTab}</h1>
          </div>
          <div className="topbar-right">
            <span className="system-status"><i /> Backend connected when available</span>
            <div className="top-avatar">CS</div>
          </div>
        </header>

        <div className="page-content">
          {error && <div className="alert error-alert">{error}</div>}
          {success && (
            <div className="alert success-alert">
              {success}
              <button onClick={() => setSuccess('')}>×</button>
            </div>
          )}

          {activeTab === 'Dashboard' && (
            <>
              <section className="welcome-banner">
                <div>
                  <span className="eyebrow">CYBERSECURITY INTELLIGENCE</span>
                  <h2>Your security knowledge, <span>amplified.</span></h2>
                  <p>
                    Analyze cybersecurity documents, explore vulnerabilities,
                    and get AI-powered security insights.
                  </p>
                  <button
                    className="primary-button"
                    onClick={() => setActiveTab('AI Assistant')}
                  >
                    Open AI Assistant <span>→</span>
                  </button>
                </div>
                <div className="banner-art">
                  <div className="orbit orbit-one" />
                  <div className="orbit orbit-two" />
                  <div className="shield">⬡</div>
                  <span className="orbit-dot dot-one" />
                  <span className="orbit-dot dot-two" />
                  <span className="orbit-dot dot-three" />
                </div>
              </section>

              <section className="section-heading">
                <div>
                  <h2>Knowledge overview</h2>
                  <p>Your indexed cybersecurity intelligence</p>
                </div>
                <button className="refresh-button" onClick={loadDashboard}>
                  ↻ Refresh
                </button>
              </section>

              <section className="stats-grid">
                <StatCard
                  icon="▤"
                  label="Total documents"
                  value={stats.total_documents ?? 0}
                  detail="Indexed knowledge files"
                />
                <StatCard
                  icon="◈"
                  label="Knowledge chunks"
                  value={stats.total_chunks ?? 0}
                  detail="Searchable text segments"
                />
                <StatCard
                  icon="✧"
                  label="AI engine"
                  value="RAG"
                  detail="Retrieval augmented generation"
                />
              </section>

              <section className="content-grid">
                <div className="panel">
                  <div className="panel-heading">
                    <div>
                      <h3>Recent documents</h3>
                      <p>Your latest indexed files</p>
                    </div>
                    <button
                      className="text-button"
                      onClick={() => setActiveTab('Documents')}
                    >
                      View all →
                    </button>
                  </div>

                  {documents.length === 0 ? (
                    <EmptyState
                      title="No documents yet"
                      description="Upload your first cybersecurity document to get started."
                    />
                  ) : (
                    <div className="document-list">
                      {documents.slice(0, 5).map((doc) => (
                        <DocumentRow key={doc.document_id} doc={doc} />
                      ))}
                    </div>
                  )}
                </div>

                <div className="panel quick-panel">
                  <div className="panel-heading">
                    <div>
                      <h3>Quick actions</h3>
                      <p>Continue your security workflow</p>
                    </div>
                  </div>
                  <button
                    className="quick-action"
                    onClick={() => setActiveTab('Documents')}
                  >
                    <span className="action-icon">＋</span>
                    <span><strong>Upload document</strong><small>Index PDF, TXT or DOCX files</small></span>
                    <b>→</b>
                  </button>
                  <button
                    className="quick-action"
                    onClick={() => setActiveTab('AI Assistant')}
                  >
                    <span className="action-icon purple">✧</span>
                    <span><strong>Ask the AI assistant</strong><small>Explore your security knowledge</small></span>
                    <b>→</b>
                  </button>
                </div>
              </section>
            </>
          )}

          {activeTab === 'Documents' && (
            <section className="documents-page">
              <div className="upload-panel">
                <div className="upload-symbol">⇧</div>
                <h2>Build your security knowledge base</h2>
                <p>
                  Upload security reports, OWASP references, and technical
                  documents for AI-powered analysis.
                </p>
                <button
                  className="primary-button"
                  onClick={() => fileInput.current?.click()}
                  disabled={uploading}
                >
                  {uploading ? 'Uploading...' : '＋ Upload document'}
                </button>
                <input
                  ref={fileInput}
                  type="file"
                  accept=".pdf,.txt,.docx"
                  hidden
                  onChange={handleUpload}
                />
                <span className="upload-hint">PDF, TXT, DOCX · Maximum 10 MB</span>
              </div>

              <div className="panel documents-table-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Indexed documents</h3>
                    <p>{documents.length} documents in your knowledge base</p>
                  </div>
                  <button className="refresh-button" onClick={loadDashboard}>
                    ↻ Refresh
                  </button>
                </div>

                {documents.length === 0 ? (
                  <EmptyState
                    title="Your knowledge base is empty"
                    description="Upload a document to start building your cybersecurity knowledge base."
                  />
                ) : (
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>DOCUMENT</th>
                          <th>PAGES</th>
                          <th>CHUNKS</th>
                          <th>STATUS</th>
                          <th>ACTION</th>
                        </tr>
                      </thead>
                      <tbody>
                        {documents.map((doc) => (
                          <tr key={doc.document_id}>
                            <td>
                              <div className="file-cell">
                                <span className="file-icon">▤</span>
                                <strong>{doc.filename}</strong>
                              </div>
                            </td>
                            <td>{doc.pages ?? '—'}</td>
                            <td>{doc.chunks ?? 0}</td>
                            <td><span className="status-pill">Indexed</span></td>
                            <td>
                              <button
                                className="delete-button"
                                onClick={() => handleDelete(doc.document_id)}
                              >
                                Delete
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </section>
          )}

          {activeTab === 'AI Assistant' && (
            <section className="chat-panel">
              <div className="chat-heading">
                <div className="assistant-avatar">✧</div>
                <div>
                  <h2>CyberShield AI Assistant</h2>
                  <p><i /> Ask questions about your indexed documents</p>
                </div>
                <span className="rag-badge">RAG ENABLED</span>
              </div>

              <div className="chat-messages">
                {messages.map((message, index) => (
                  <div className={`message ${message.role}`} key={index}>
                    <div className="message-avatar">
                      {message.role === 'assistant' ? '✧' : 'CS'}
                    </div>
                    <div className="message-body">
                      <span className="message-author">
                        {message.role === 'assistant' ? 'CyberShield AI' : 'You'}
                      </span>
                      <div className="message-bubble">{message.content}</div>
                      
{message.sources?.length > 0 && (
  <div className="sources">
    <strong>Retrieved knowledge sources</strong>

    {message.sources.map((source, i) => (
      <div className="source-item" key={source.source_id || i}>
        <div className="source-header">
          <strong>
            {source.filename || source.source || `Source ${i + 1}`}
          </strong>

          <span>
            {source.page_number
              ? `Page ${source.page_number}`
              : 'Document source'}
          </span>
        </div>

        {typeof source.distance === 'number' && (
          <div className="source-relevance">
            Relevance distance: {source.distance.toFixed(3)}
          </div>
        )}

        {source.content_preview && (
          <p className="source-preview">
            {source.content_preview}
          </p>
        )}
      </div>
    ))}
  </div>
)}
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="message assistant">
                    <div className="message-avatar">✧</div>
                    <div className="message-body">
                      <span className="message-author">CyberShield AI</span>
                      <div className="message-bubble typing">Analyzing your knowledge base...</div>
                    </div>
                  </div>
                )}
                <div ref={chatEnd} />
              </div>

              <form className="chat-input-area" onSubmit={handleAsk}>
                <input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Ask about SQL injection, IDOR, XSS, OWASP..."
                  disabled={loading}
                />
                <button type="submit" disabled={loading || !question.trim()}>
                  {loading ? '...' : '↑'}
                </button>
              </form>
              <p className="chat-disclaimer">
                AI-generated responses can contain errors. Verify important security findings.
              </p>
            </section>
          )}
        </div>

        <footer className="footer">
          <span>© 2026 CyberShield AI</span>
          <span>Intelligent Cybersecurity RAG Assistant</span>
        </footer>
      </main>
    </div>
  )
}

function StatCard({ icon, label, value, detail }) {
  return (
    <div className="stat-card">
      <div className="stat-top">
        <span>{label}</span>
        <span className="stat-icon">{icon}</span>
      </div>
      <strong className="stat-value">{value}</strong>
      <span className="stat-detail">{detail}</span>
    </div>
  )
}

function DocumentRow({ doc }) {
  return (
    <div className="document-row">
      <span className="file-icon">▤</span>
      <div className="document-info">
        <strong>{doc.filename}</strong>
        <span>{doc.chunks} chunks · {doc.pages} pages</span>
      </div>
      <span className="status-pill">Indexed</span>
    </div>
  )
}

function EmptyState({ title, description }) {
  return (
    <div className="empty-state">
      <div>▤</div>
      <strong>{title}</strong>
      <p>{description}</p>
    </div>
  )
}

export default App
