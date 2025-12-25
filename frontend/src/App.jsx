import React, { useState, useEffect } from 'react'
import PDFViewer from './PDFViewer'
import './App.css'

const API = '/api'

function App() {
  const [tab, setTab] = useState('scrape')

  return (
    <div className="app">
      <header>
        <h1>PDF Redaction Testing Tool</h1>
        <p>Detect improperly redacted PDFs</p>
      </header>

      <nav>
        <button className={tab === 'scrape' ? 'active' : ''} onClick={() => setTab('scrape')}>Scrape URLs</button>
        <button className={tab === 'lists' ? 'active' : ''} onClick={() => setTab('lists')}>URL Lists</button>
        <button className={tab === 'jobs' ? 'active' : ''} onClick={() => setTab('jobs')}>Jobs</button>
        <button className={tab === 'results' ? 'active' : ''} onClick={() => setTab('results')}>Results</button>
        <button className={tab === 'upload' ? 'active' : ''} onClick={() => setTab('upload')}>Upload</button>
      </nav>

      <main>
        {tab === 'scrape' && <ScrapePanel />}
        {tab === 'lists' && <UrlListsPanel />}
        {tab === 'jobs' && <JobsPanel />}
        {tab === 'results' && <ResultsPanel />}
        {tab === 'upload' && <UploadPanel />}
      </main>
    </div>
  )
}

function ScrapePanel() {
  const [url, setUrl] = useState('')
  const [name, setName] = useState('')
  const [useBrightdata, setUseBrightdata] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleScrape = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API}/scrape-urls`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, name, use_brightdata: useBrightdata })
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Failed to scrape')
      setResult(data)
    } catch (e) {
      setError(e.message)
    }
    setLoading(false)
  }

  return (
    <div className="panel">
      <h2>Scrape URL for PDFs</h2>
      <div className="form-group">
        <label>URL to scrape:</label>
        <input
          type="url"
          value={url}
          onChange={e => setUrl(e.target.value)}
          placeholder="https://example.com/documents"
        />
      </div>
      <div className="form-group">
        <label>List name:</label>
        <input
          type="text"
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder="my-document-list"
        />
      </div>
      <div className="form-group checkbox">
        <label>
          <input
            type="checkbox"
            checked={useBrightdata}
            onChange={e => setUseBrightdata(e.target.checked)}
          />
          Use Bright Data proxy (for protected sites)
        </label>
      </div>
      <button onClick={handleScrape} disabled={loading || !url || !name}>
        {loading ? 'Scraping...' : 'Scrape URLs'}
      </button>

      {error && <div className="error">{error}</div>}
      {result && (
        <div className="result">
          <h3>Found {result.total_urls} PDFs</h3>
          <p>Saved as: {result.name}</p>
          <div className="url-preview">
            {result.urls?.slice(0, 5).map((u, i) => (
              <div key={i} className="url-item">{decodeURIComponent(u.split('/').pop())}</div>
            ))}
            {result.total_urls > 5 && <div className="url-item">...and {result.total_urls - 5} more</div>}
          </div>
        </div>
      )}
    </div>
  )
}

function UrlListsPanel() {
  const [lists, setLists] = useState([])
  const [loading, setLoading] = useState(true)
  const [downloading, setDownloading] = useState(null)
  const [downloadResult, setDownloadResult] = useState(null)

  useEffect(() => {
    fetchLists()
  }, [])

  const fetchLists = async () => {
    try {
      const res = await fetch(`${API}/url-lists`)
      const data = await res.json()
      setLists(data.lists || [])
    } catch (e) {
      console.error(e)
    }
    setLoading(false)
  }

  const startDownload = async (name, totalUrls) => {
    setDownloading(name)
    try {
      const res = await fetch(`${API}/download-async`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          max_concurrent: 20,
          max_pdfs: totalUrls,
          analyze_after: true
        })
      })
      const data = await res.json()
      setDownloadResult(data)
    } catch (e) {
      console.error(e)
    }
    setDownloading(null)
  }

  if (loading) return <div className="panel"><p>Loading...</p></div>

  return (
    <div className="panel">
      <h2>Stored URL Lists</h2>
      {downloadResult && (
        <div className="result">
          <p>Job started: {downloadResult.job_id}</p>
          <p>{downloadResult.message}</p>
        </div>
      )}
      <div className="lists">
        {lists.length === 0 ? (
          <p>No URL lists. Scrape a URL first.</p>
        ) : (
          lists.map(list => (
            <div key={list.name} className="list-item">
              <div className="list-info">
                <strong>{list.name}</strong>
                <span>{list.total_urls} PDFs</span>
                <small>{list.source_url}</small>
              </div>
              <button
                onClick={() => startDownload(list.name, list.total_urls)}
                disabled={downloading === list.name}
              >
                {downloading === list.name ? 'Starting...' : 'Download & Analyze'}
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

function JobsPanel() {
  const [jobs, setJobs] = useState([])
  const [selectedJob, setSelectedJob] = useState(null)
  const [jobDetail, setJobDetail] = useState(null)

  useEffect(() => {
    fetchJobs()
    const interval = setInterval(fetchJobs, 5000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (selectedJob) {
      fetchJobDetail(selectedJob)
      const interval = setInterval(() => fetchJobDetail(selectedJob), 2000)
      return () => clearInterval(interval)
    }
  }, [selectedJob])

  const fetchJobs = async () => {
    try {
      const res = await fetch(`${API}/jobs`)
      const data = await res.json()
      setJobs(data.jobs || [])
    } catch (e) {
      console.error(e)
    }
  }

  const fetchJobDetail = async (jobId) => {
    try {
      const res = await fetch(`${API}/jobs/${jobId}`)
      const data = await res.json()
      setJobDetail(data)
    } catch (e) {
      console.error(e)
    }
  }

  return (
    <div className="panel">
      <h2>Download & Analysis Jobs</h2>
      <div className="jobs-container">
        <div className="jobs-list">
          {jobs.length === 0 ? (
            <p>No jobs yet.</p>
          ) : (
            jobs.map(job => (
              <div
                key={job.job_id}
                className={`job-item ${selectedJob === job.job_id ? 'selected' : ''} ${job.status}`}
                onClick={() => setSelectedJob(job.job_id)}
              >
                <div className="job-status">{job.status}</div>
                <div className="job-progress">
                  {job.downloaded}/{job.total} downloaded
                  {job.analyzed > 0 && `, ${job.analyzed} analyzed`}
                </div>
                <small>{job.job_id.slice(0, 8)}...</small>
              </div>
            ))
          )}
        </div>

        {jobDetail && (
          <div className="job-detail">
            <h3>Job Details</h3>
            <div className="stats-grid">
              <div className="stat">
                <span className="stat-value">{jobDetail.downloaded}</span>
                <span className="stat-label">Downloaded</span>
              </div>
              <div className="stat">
                <span className="stat-value">{jobDetail.failed}</span>
                <span className="stat-label">Failed</span>
              </div>
              <div className="stat">
                <span className="stat-value">{jobDetail.analyzed}</span>
                <span className="stat-label">Analyzed</span>
              </div>
              <div className="stat vulnerable">
                <span className="stat-value">{jobDetail.vulnerable_count}</span>
                <span className="stat-label">Vulnerable</span>
              </div>
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${(jobDetail.downloaded + jobDetail.failed) / jobDetail.total * 100}%` }}
              />
            </div>
            <p className="job-phase">Phase: {jobDetail.phase} | Status: {jobDetail.status}</p>
            {jobDetail.errors?.length > 0 && (
              <details>
                <summary>{jobDetail.errors.length} errors</summary>
                <div className="errors-list">
                  {jobDetail.errors.slice(0, 10).map((e, i) => (
                    <div key={i} className="error-item">{e}</div>
                  ))}
                </div>
              </details>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function ResultsPanel() {
  const [results, setResults] = useState([])
  const [pagination, setPagination] = useState({ total: 0, page: 1, pages: 1, per_page: 20 })
  const [loading, setLoading] = useState(true)
  const [vulnerableOnly, setVulnerableOnly] = useState(true)
  const [perPage, setPerPage] = useState(20)
  const [selectedResult, setSelectedResult] = useState(null)
  const pdfViewerRef = React.useRef(null)

  useEffect(() => {
    fetchResults(1)
  }, [vulnerableOnly, perPage])

  const fetchResults = async (page) => {
    setLoading(true)
    try {
      const res = await fetch(`${API}/results?page=${page}&per_page=${perPage}&vulnerable_only=${vulnerableOnly}`)
      const data = await res.json()
      setResults(data.results || [])
      setPagination({ total: data.total, page: data.page, pages: data.pages, per_page: data.per_page })
    } catch (e) {
      console.error(e)
    }
    setLoading(false)
  }

  return (
    <div className="panel">
      <h2>Analysis Results</h2>
      <div className="filters">
        <label>
          <input
            type="checkbox"
            checked={vulnerableOnly}
            onChange={e => setVulnerableOnly(e.target.checked)}
          />
          Vulnerable only
        </label>
        <label className="per-page-select">
          Show:
          <select value={perPage} onChange={e => setPerPage(Number(e.target.value))}>
            <option value={10}>10</option>
            <option value={20}>20</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
          per page
        </label>
        <span className="total">Total: {pagination.total} results</span>
      </div>

      {loading ? (
        <p>Loading...</p>
      ) : (
        <>
          <div className="results-grid">
            {results.map(r => (
              <div
                key={r.id}
                className={`result-card ${r.is_vulnerable ? 'vulnerable' : 'safe'}`}
                onClick={() => setSelectedResult(r)}
              >
                <div className="result-filename">{r.filename}</div>
                <div className="result-status">
                  {r.is_vulnerable ? `${r.findings?.length || 0} findings` : 'Safe'}
                </div>
              </div>
            ))}
          </div>

          <div className="pagination">
            <button
              disabled={pagination.page <= 1}
              onClick={() => fetchResults(pagination.page - 1)}
            >
              Previous
            </button>
            <span>Page {pagination.page} of {pagination.pages}</span>
            <button
              disabled={pagination.page >= pagination.pages}
              onClick={() => fetchResults(pagination.page + 1)}
            >
              Next
            </button>
          </div>
        </>
      )}

      {selectedResult && (
        <div className="modal" onClick={() => setSelectedResult(null)}>
          <div className="modal-content modal-large" onClick={e => e.stopPropagation()}>
            <button className="close" onClick={() => setSelectedResult(null)}>&times;</button>
            <h3>{selectedResult.filename}</h3>
            <p className={selectedResult.is_vulnerable ? 'vulnerable' : 'safe'}>
              {selectedResult.is_vulnerable ? 'VULNERABLE' : 'Safe'}
            </p>

            {selectedResult.source_url && (
              <div className="source-url">
                <strong>Original URL:</strong>{' '}
                <a href={selectedResult.source_url} target="_blank" rel="noopener noreferrer">
                  {selectedResult.source_url}
                </a>
              </div>
            )}

            {selectedResult.findings?.length > 0 && (
              <div className="findings">
                <h4>Findings (by page):</h4>
                {selectedResult.findings.map((f, i) => (
                  <div key={i} className="finding">{f}</div>
                ))}
              </div>
            )}

            {selectedResult.potential_fake_redactions?.length > 0 && (
              <div className="redactions-detail">
                <h4>Fake Redactions Detail (click page to jump):</h4>
                {selectedResult.potential_fake_redactions.map((r, i) => (
                  <div key={i} className="redaction-item">
                    <span
                      className="page-badge clickable"
                      onClick={() => pdfViewerRef.current?.goToPage(r.page)}
                    >
                      Page {r.page}
                    </span>
                    <span className="hidden-text">{r.hidden_text}</span>
                    <span className="redaction-type">({r.type})</span>
                  </div>
                ))}
              </div>
            )}

            {selectedResult.file_id && (
              <div className="pdf-viewer">
                <h4>PDF Document (yellow highlights show recovered text):</h4>
                <PDFViewer
                  ref={pdfViewerRef}
                  fileId={selectedResult.file_id}
                  redactions={selectedResult.potential_fake_redactions || []}
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function UploadPanel() {
  const [files, setFiles] = useState([])
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)

  const handleUpload = async () => {
    if (files.length === 0) return
    setUploading(true)

    const formData = new FormData()
    for (const file of files) {
      formData.append('files', file)
    }

    try {
      const res = await fetch(`${API}/upload`, {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      setResult(data)

      // Trigger analysis
      if (data.files?.length > 0) {
        await fetch(`${API}/analyze-uploaded`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ persist: true })
        })
      }
    } catch (e) {
      console.error(e)
    }
    setUploading(false)
  }

  return (
    <div className="panel">
      <h2>Upload PDFs</h2>
      <div className="upload-area">
        <input
          type="file"
          multiple
          accept=".pdf"
          onChange={e => setFiles(Array.from(e.target.files))}
        />
        <p>{files.length} file(s) selected</p>
        <button onClick={handleUpload} disabled={uploading || files.length === 0}>
          {uploading ? 'Uploading...' : 'Upload & Analyze'}
        </button>
      </div>
      {result && (
        <div className="result">
          <p>Uploaded {result.uploaded} files</p>
        </div>
      )}
    </div>
  )
}

export default App
