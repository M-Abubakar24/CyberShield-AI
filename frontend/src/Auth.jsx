
import { useState } from 'react'
import './Auth.css'

const API_BASE = 'http://127.0.0.1:8000'

function Auth({ onAuthenticated }) {
  const [mode, setMode] = useState('login')
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  const handleSubmit = async (event) => {
    event.preventDefault()
    setError('')
    setSuccess('')
    setLoading(true)

    try {
      if (mode === 'register') {
        const registerResponse = await fetch(
          `${API_BASE}/api/auth/register`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              full_name: fullName.trim(),
              email: email.trim(),
              password,
            }),
          }
        )

        const registerData = await registerResponse.json()

        if (!registerResponse.ok) {
          throw new Error(
            registerData.detail || 'Registration failed.'
          )
        }

        setSuccess('Account created successfully. Please sign in.')
        setMode('login')
        setPassword('')
        return
      }

      const formData = new URLSearchParams()
      formData.append('username', email.trim())
      formData.append('password', password)

      const loginResponse = await fetch(
        `${API_BASE}/api/auth/login`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: formData,
        }
      )

      const loginData = await loginResponse.json()

      if (!loginResponse.ok) {
        throw new Error(loginData.detail || 'Login failed.')
      }

      const profileResponse = await fetch(
        `${API_BASE}/api/auth/me`,
        {
          headers: {
            Authorization: `Bearer ${loginData.access_token}`,
          },
        }
      )

      const profileData = await profileResponse.json()

      if (!profileResponse.ok) {
        throw new Error('Could not load your user profile.')
      }

      onAuthenticated(loginData.access_token, profileData)
    } catch (err) {
      setError(err.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-screen">
      <div className="auth-card">
        <div className="auth-brand-icon">⌘</div>

        <h1>CyberShield <span>AI</span></h1>
        <p className="auth-subtitle">
          Intelligent Cybersecurity RAG Assistant
        </p>

        <div className="auth-heading">
          <h2>
            {mode === 'login' ? 'Welcome back' : 'Create your account'}
          </h2>
          <p>
            {mode === 'login'
              ? 'Sign in to access your security workspace.'
              : 'Register to create your personal security workspace.'}
          </p>
        </div>

        {error && <div className="auth-error">{error}</div>}
        {success && <div className="auth-success">{success}</div>}

        <form onSubmit={handleSubmit}>
          {mode === 'register' && (
            <label>
              Full name
              <input
                type="text"
                value={fullName}
                onChange={(event) => setFullName(event.target.value)}
                placeholder="Enter your full name"
                minLength={2}
                maxLength={100}
                required
              />
            </label>
          )}

          <label>
            Email address
            <input
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              required
            />
          </label>

          <label>
            Password
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter your password"
              minLength={mode === 'register' ? 8 : undefined}
              maxLength={128}
              required
            />
          </label>

          <button
            className="auth-submit"
            type="submit"
            disabled={loading}
          >
            {loading
              ? 'Please wait...'
              : mode === 'login'
                ? 'Sign in'
                : 'Create account'}
          </button>
        </form>

        <div className="auth-switch">
          {mode === 'login'
            ? "Don't have an account?"
            : 'Already have an account?'}

          <button
            type="button"
            onClick={() => {
              setMode(mode === 'login' ? 'register' : 'login')
              setError('')
              setSuccess('')
            }}
          >
            {mode === 'login' ? 'Create account' : 'Sign in'}
          </button>
        </div>

        <div className="auth-footer">
          Protected cybersecurity intelligence workspace
        </div>
      </div>
    </div>
  )
}

export default Auth
