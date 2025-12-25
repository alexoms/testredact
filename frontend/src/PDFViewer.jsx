import React, { useState, forwardRef, useImperativeHandle } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/esm/Page/AnnotationLayer.css'
import 'react-pdf/dist/esm/Page/TextLayer.css'

// Configure worker - use a known working version
pdfjs.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js'

const PDFViewer = forwardRef(({ fileId, redactions = [] }, ref) => {
  const [numPages, setNumPages] = useState(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [scale, setScale] = useState(1.0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  if (!fileId) {
    return <div className="pdf-no-file">No PDF file available</div>
  }

  // Group redactions by page
  const redactionsByPage = redactions.reduce((acc, r) => {
    const page = r.page || 1
    if (!acc[page]) acc[page] = []
    acc[page].push(r)
    return acc
  }, {})

  const pagesWithRedactions = Object.keys(redactionsByPage).map(Number).sort((a, b) => a - b)
  const currentRedactions = redactionsByPage[currentPage] || []

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages)
    setLoading(false)
    // Jump to first page with findings
    if (pagesWithRedactions.length > 0) {
      setCurrentPage(pagesWithRedactions[0])
    }
  }

  const onDocumentLoadError = (err) => {
    setError(err.message)
    setLoading(false)
  }

  const goToPage = (page) => {
    if (page >= 1 && (!numPages || page <= numPages)) {
      setCurrentPage(page)
    }
  }

  // Expose goToPage to parent via ref
  useImperativeHandle(ref, () => ({
    goToPage
  }))

  return (
    <div className="pdf-viewer-container">
      {pagesWithRedactions.length > 0 && (
        <div className="pdf-toolbar">
          <div className="pdf-redaction-nav">
            <span className="findings-label">Jump to findings on page:</span>
            {pagesWithRedactions.map(page => (
              <button
                key={page}
                className={`page-jump ${currentPage === page ? 'active' : ''}`}
                onClick={() => goToPage(page)}
              >
                Page {page}
                <span className="redaction-count">
                  ({redactionsByPage[page].length})
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {currentRedactions.length > 0 && (
        <div className="current-page-findings">
          <div className="findings-header">
            Hidden text found on page {currentPage}:
          </div>
          {currentRedactions.map((r, i) => (
            <div key={i} className="finding-highlight">
              <span className="finding-number">{i + 1}</span>
              <span className="finding-text">{r.hidden_text}</span>
              <span className="finding-type">{r.type}</span>
            </div>
          ))}
        </div>
      )}

      <div className="pdf-controls">
        <div className="pdf-nav">
          <button onClick={() => goToPage(1)} disabled={currentPage === 1}>First</button>
          <button onClick={() => goToPage(currentPage - 1)} disabled={currentPage === 1}>Prev</button>
          <span className="page-info">
            Page {currentPage} of {numPages || '?'}
          </span>
          <button onClick={() => goToPage(currentPage + 1)} disabled={currentPage === numPages}>Next</button>
          <button onClick={() => goToPage(numPages)} disabled={currentPage === numPages}>Last</button>
        </div>
        <div className="pdf-zoom">
          <button onClick={() => setScale(s => Math.max(0.5, s - 0.25))}>-</button>
          <span>{Math.round(scale * 100)}%</span>
          <button onClick={() => setScale(s => Math.min(2.5, s + 0.25))}>+</button>
        </div>
      </div>

      <div className="pdf-document-container">
        {loading && <div className="pdf-loading">Loading PDF...</div>}
        {error && (
          <div className="pdf-error">
            <p>Error loading PDF: {error}</p>
            <a
              href={`/api/uploads/${fileId}?highlight=true`}
              target="_blank"
              rel="noopener noreferrer"
              className="pdf-download-link"
            >
              Open in new tab instead
            </a>
          </div>
        )}
        <Document
          file={`/api/uploads/${fileId}?highlight=true`}
          onLoadSuccess={onDocumentLoadSuccess}
          onLoadError={onDocumentLoadError}
          loading=""
        >
          <Page
            pageNumber={currentPage}
            scale={scale}
            renderTextLayer={true}
            renderAnnotationLayer={true}
          />
        </Document>
      </div>

      <div className="pdf-actions">
        <a
          href={`/api/uploads/${fileId}?highlight=true`}
          target="_blank"
          rel="noopener noreferrer"
          className="open-pdf-btn"
        >
          Open Highlighted PDF in New Tab
        </a>
        <a
          href={`/api/uploads/${fileId}`}
          target="_blank"
          rel="noopener noreferrer"
          className="open-pdf-btn secondary"
        >
          Open Original PDF
        </a>
      </div>
    </div>
  )
})

export default PDFViewer
