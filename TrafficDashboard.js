"use client"

import { useState, useEffect } from "react"
import axios from "axios"
import "./style.css"

const TrafficDashboard = () => {
  const [dashboardData, setDashboardData] = useState({
    total_violations: 0,
    total_vehicles: 0,
    total_videos: 0,
    pending_videos: 0,
    recent_violations: [],
    violation_trends: [],
    detailed_stats: {
      total_fines: 0,
      unique_license_plates: 0,
      avg_confidence: 0,
      high_severity: 0,
      medium_severity: 0,
      low_severity: 0
    }
  })

  const [violationSummary, setViolationSummary] = useState([])
  const [allViolations, setAllViolations] = useState([])
  const [detailedStats, setDetailedStats] = useState(null)
  const [videos, setVideos] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    const fetchData = async () => {
      try {
        await Promise.all([
          fetchDashboardData(),
          fetchViolationSummary(),
          fetchAllViolations(),
          fetchDetailedStats(),
          fetchVideos()
        ])
        setError(null)
      } catch (err) {
        console.error("Error fetching data:", err)
        setError("Failed to load data")
      } finally {
        setLoading(false)
      }
    }

    // Initial data fetch
    fetchData()

    // Refresh data every 30 seconds
    const interval = setInterval(() => {
      fetchData()
    }, 30000)

    return () => clearInterval(interval)
  }, []) // Empty dependency array is correct here

  const fetchAllData = async () => {
    try {
      await Promise.all([
        fetchDashboardData(),
        fetchViolationSummary(),
        fetchAllViolations(),
        fetchDetailedStats(),
        fetchVideos()
      ])
      setError(null)
    } catch (err) {
      console.error("Error fetching data:", err)
      setError("Failed to load data")
    } finally {
      setLoading(false)
    }
  }

  const fetchDashboardData = async () => {
    const response = await axios.get("http://localhost:8000/api/dashboard/")
    setDashboardData(response.data)
  }

  const fetchViolationSummary = async () => {
    const response = await axios.get("http://localhost:8000/api/violation-summary/")
    setViolationSummary(response.data)
  }

  const fetchAllViolations = async () => {
    const response = await axios.get("http://localhost:8000/api/violations/all")
    setAllViolations(response.data.violations || [])
  }

  const fetchDetailedStats = async () => {
    const response = await axios.get("http://localhost:8000/api/statistics/detailed")
    setDetailedStats(response.data)
  }

  const fetchVideos = async () => {
    const response = await axios.get("http://localhost:8000/api/videos/")
    setVideos(response.data)
  }

  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0])
  }

  const handleVideoUpload = async () => {
    if (!selectedFile) {
      alert("Please select a video file first")
      return
    }

    setUploading(true)
    const formData = new FormData()
    formData.append("video", selectedFile)

    try {
      const response = await axios.post("http://localhost:8000/api/upload-video/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })

      alert(`Video uploaded successfully! Processing will begin shortly.\nFile: ${response.data.original_name}\nSize: ${response.data.file_size_mb} MB`)
      setSelectedFile(null)
      document.getElementById("video-input").value = ""

      // Refresh data immediately and then again after 5 seconds
      fetchAllData()
      setTimeout(() => {
        fetchAllData()
      }, 5000)
    } catch (err) {
      console.error("Error uploading video:", err)
      alert("Failed to upload video. Please try again.")
    } finally {
      setUploading(false)
    }
  }

  const formatViolationType = (type) => {
    return type.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString()
  }

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount || 0)
  }

  if (loading) {
    return (
      <div className="dashboard-container">
        <div className="loading">Loading enhanced dashboard...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="dashboard-container">
        <div className="error">Error: {error}</div>
      </div>
    )
  }

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>Enhanced Traffic Violation Detection System</h1>
        <p>Real-time monitoring with AI-powered vehicle detection and license plate recognition</p>
        <div className="system-status">
          <span className="status-indicator active"></span>
          System Active • Last Updated: {new Date().toLocaleTimeString()}
        </div>
      </header>

      {/* Enhanced Stats Cards */}
      <div className="stats-grid">
        <div className="stat-card violations">
          <div className="stat-icon">🚨</div>
          <div className="stat-content">
            <h3>Total Violations</h3>
            <p className="stat-number">{dashboardData.total_violations}</p>
            <p className="stat-subtitle">
              {dashboardData.detailed_stats.high_severity} High • {dashboardData.detailed_stats.medium_severity} Medium • {dashboardData.detailed_stats.low_severity} Low
            </p>
          </div>
        </div>

        <div className="stat-card vehicles">
          <div className="stat-icon">🚗</div>
          <div className="stat-content">
            <h3>Detected Vehicles</h3>
            <p className="stat-number">{dashboardData.total_vehicles}</p>
            <p className="stat-subtitle">{dashboardData.detailed_stats.unique_license_plates} with license plates</p>
          </div>
        </div>

        <div className="stat-card revenue">
          <div className="stat-icon">💰</div>
          <div className="stat-content">
            <h3>Total Fines</h3>
            <p className="stat-number">{formatCurrency(dashboardData.detailed_stats.total_fines)}</p>
            <p className="stat-subtitle">Revenue generated</p>
          </div>
        </div>

        <div className="stat-card accuracy">
          <div className="stat-icon">🎯</div>
          <div className="stat-content">
            <h3>Detection Accuracy</h3>
            <p className="stat-number">{(dashboardData.detailed_stats.avg_confidence * 100).toFixed(1)}%</p>
            <p className="stat-subtitle">Average confidence score</p>
          </div>
        </div>

        <div className="stat-card videos">
          <div className="stat-icon">📹</div>
          <div className="stat-content">
            <h3>Videos Processed</h3>
            <p className="stat-number">{dashboardData.total_videos}</p>
            <p className="stat-subtitle">{dashboardData.pending_videos} pending</p>
          </div>
        </div>
      </div>

      {/* Video Upload Section */}
      <div className="upload-section">
        <h2>Upload Video for AI Analysis</h2>
        <div className="upload-container">
          <input 
            id="video-input" 
            type="file" 
            accept="video/*" 
            onChange={handleFileSelect} 
            className="file-input" 
          />
          <button 
            onClick={handleVideoUpload} 
            disabled={uploading || !selectedFile} 
            className="upload-button"
          >
            {uploading ? "🔄 Processing..." : "📤 Upload & Analyze"}
          </button>
        </div>
        {selectedFile && (
          <div className="selected-file">
            📁 Selected: {selectedFile.name} ({(selectedFile.size / (1024*1024)).toFixed(2)} MB)
          </div>
        )}
      </div>

      {/* Navigation Tabs */}
      <div className="tab-navigation">
        <button 
          className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          📊 Overview
        </button>
        <button 
          className={`tab-button ${activeTab === 'violations' ? 'active' : ''}`}
          onClick={() => setActiveTab('violations')}
        >
          🚨 All Violations ({allViolations.length})
        </button>
        <button 
          className={`tab-button ${activeTab === 'analytics' ? 'active' : ''}`}
          onClick={() => setActiveTab('analytics')}
        >
          📈 Analytics
        </button>
        <button 
          className={`tab-button ${activeTab === 'videos' ? 'active' : ''}`}
          onClick={() => setActiveTab('videos')}
        >
          📹 Videos ({videos.length})
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="tab-content">
          <div className="charts-section">
            <div className="chart-container">
              <h2>Violation Types Distribution</h2>
              <div className="chart-content">
                {violationSummary.length > 0 ? (
                  <div className="bar-chart">
                    {violationSummary.map((item, index) => (
                      <div key={index} className="bar-item">
                        <div className="bar-label">
                          {formatViolationType(item.violation_type)}
                          <span className="confidence-score">
                            (Avg: {(item.avg_confidence * 100).toFixed(1)}%)
                          </span>
                        </div>
                        <div className="bar-wrapper">
                          <div
                            className="bar-fill"
                            style={{
                              width: `${(item.count / Math.max(...violationSummary.map((v) => v.count))) * 100}%`,
                            }}
                          ></div>
                          <span className="bar-value">
                            {item.count} • {formatCurrency(item.total_fines)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="no-data">No violation data available</p>
                )}
              </div>
            </div>

            <div className="chart-container">
              <h2>Recent Activity</h2>
              <div className="chart-content">
                {dashboardData.recent_violations.length > 0 ? (
                  <div className="activity-list">
                    {dashboardData.recent_violations.slice(0, 5).map((violation, index) => (
                      <div key={index} className="activity-item">
                        <div className="activity-icon">🚨</div>
                        <div className="activity-details">
                          <p className="activity-title">
                            {formatViolationType(violation.violation_type)}
                            {violation.speed && ` (${violation.speed.toFixed(1)} km/h)`}
                          </p>
                          <p className="activity-subtitle">
                            🚗 {violation.vehicle_license_plate} • 
                            📍 {violation.location} • 
                            🎯 {(violation.confidence * 100).toFixed(1)}% confidence
                          </p>
                          <p className="activity-time">{formatDate(violation.timestamp)}</p>
                        </div>
                        <div className={`severity-badge ${violation.severity}`}>
                          {violation.severity}
                        </div>
                        <div className="fine-amount">
                          {formatCurrency(violation.fine_amount)}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="no-data">No recent violations</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'violations' && (
        <div className="tab-content">
          <div className="table-section">
            <h2>All Traffic Violations ({allViolations.length})</h2>
            <div className="table-container">
              {allViolations.length > 0 ? (
                <table className="violations-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Vehicle</th>
                      <th>Type</th>
                      <th>Violation Type</th>
                      <th>Speed</th>
                      <th>Location</th>
                      <th>Confidence</th>
                      <th>Severity</th>
                      <th>Fine</th>
                      <th>Timestamp</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {allViolations.map((violation, index) => (
                      <tr key={violation.id}>
                        <td>#{violation.id}</td>
                        <td>
                          <div className="vehicle-info">
                            <strong>{violation.vehicle_license_plate}</strong>
                            <br />
                            <small>{violation.vehicle_type}</small>
                          </div>
                        </td>
                        <td>{violation.vehicle_type}</td>
                        <td>{formatViolationType(violation.violation_type)}</td>
                        <td>{violation.speed ? `${violation.speed.toFixed(1)} km/h` : '-'}</td>
                        <td>{violation.location}</td>
                        <td>
                          <span className={`confidence-score ${violation.confidence > 0.8 ? 'high' : violation.confidence > 0.6 ? 'medium' : 'low'}`}>
                            {(violation.confidence * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td>
                          <span className={`severity-badge ${violation.severity}`}>
                            {violation.severity}
                          </span>
                        </td>
                        <td>{formatCurrency(violation.fine_amount)}</td>
                        <td>{formatDate(violation.timestamp)}</td>
                        <td>
                          <span className={`status-badge ${violation.is_resolved ? "resolved" : "pending"}`}>
                            {violation.is_resolved ? "Resolved" : "Pending"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="no-data">No violations recorded yet</p>
              )}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'analytics' && detailedStats && (
        <div className="tab-content">
          <div className="analytics-section">
            <h2>Detailed Analytics</h2>
            
            {/* Daily Trends */}
            <div className="chart-container">
              <h3>Daily Violation Trends (Last 30 Days)</h3>
              <div className="trend-chart">
                {detailedStats.daily_trends.slice(0, 7).map((day, index) => (
                  <div key={index} className="trend-item">
                    <div className="trend-date">{new Date(day.date).toLocaleDateString()}</div>
                    <div className="trend-stats">
                      <span>🚨 {day.violations} violations</span>
                      <span>🚗 {day.unique_vehicles} vehicles</span>
                      <span>💰 {formatCurrency(day.fines)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Repeat Offenders */}
            {detailedStats.repeat_offenders.length > 0 && (
              <div className="chart-container">
                <h3>Repeat Offenders</h3>
                <div className="offenders-list">
                  {detailedStats.repeat_offenders.slice(0, 10).map((offender, index) => (
                    <div key={index} className="offender-item">
                      <div className="offender-plate">🚗 {offender.license_plate}</div>
                      <div className="offender-stats">
                        <span>🚨 {offender.violations} violations</span>
                        <span>💰 {formatCurrency(offender.total_fines)}</span>
                        <span>📅 Last: {formatDate(offender.last_violation)}</span>
                      </div>
                      <div className="offender-types">
                        {offender.violation_types.map(type => formatViolationType(type)).join(', ')}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'videos' && (
        <div className="tab-content">
          <div className="videos-section">
            <h2>Video Processing History</h2>
            <div className="videos-grid">
              {videos.map((video, index) => (
                <div key={video.id} className="video-card">
                  <div className="video-header">
                    <h4>{video.original_name}</h4>
                    <span className={`status-badge ${video.status}`}>
                      {video.status}
                    </span>
                  </div>
                  <div className="video-stats">
                    <div className="video-stat">
                      <span className="stat-label">File Size:</span>
                      <span className="stat-value">{video.file_size_mb} MB</span>
                    </div>
                    <div className="video-stat">
                      <span className="stat-label">Violations Found:</span>
                      <span className="stat-value">{video.total_violations || 0}</span>
                    </div>
                    <div className="video-stat">
                      <span className="stat-label">Vehicles Detected:</span>
                      <span className="stat-value">{video.total_vehicles || 0}</span>
                    </div>
                    <div className="video-stat">
                      <span className="stat-label">Processing Time:</span>
                      <span className="stat-value">{video.processing_duration || 0}s</span>
                    </div>
                    <div className="video-stat">
                      <span className="stat-label">Uploaded:</span>
                      <span className="stat-value">{formatDate(video.upload_time)}</span>
                    </div>
                    {video.processed_time && (
                      <div className="video-stat">
                        <span className="stat-label">Completed:</span>
                        <span className="stat-value">{formatDate(video.processed_time)}</span>
                      </div>
                    )}
                    {video.error_message && (
                      <div className="video-error">
                        ❌ Error: {video.error_message}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default TrafficDashboard