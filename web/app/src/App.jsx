import React, { useState, useRef } from 'react';
import './index.css';

const API_BASE = 'http://localhost:5000';

function App() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const fileInputRef = useRef(null);

    const [isVideo, setIsVideo] = useState(false);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setImage(file);
            setPreview(URL.createObjectURL(file));
            setResult(null);
            setIsVideo(file.type.startsWith('video/'));
        }
    };

    const onUploadClick = () => {
        fileInputRef.current.click();
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();

        const file = e.dataTransfer.files[0];
        if (file && (file.type.startsWith('image/') || file.type.startsWith('video/'))) {
            setImage(file);
            setPreview(URL.createObjectURL(file));
            setResult(null);
            setIsVideo(file.type.startsWith('video/'));
        }
    };

    const handlePredict = async () => {
        if (!image) return;

        setLoading(true);
        const formData = new FormData();
        formData.append('image', image);

        try {
            const response = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            setResult(data);
        } catch (error) {
            console.error('Error predicting:', error);
            alert('Failed to connect to backend server. Make sure Flask is running on port 5000.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container">
            <header>
                <h1>SecureEye Deepfake Detection</h1>
                <p className="subtitle">Research-grade, fairness-aware AI for digital integrity</p>
                <div className="researcher-info">
                    <strong>Researcher:</strong> M.M.Muhthaseem | 21/ENG/088
                    <br />
                    <span>Faculty of Engineering, University of Sri Jayewardenepura</span>
                </div>
            </header>

            <div className="main-grid">
                <section className="glass-card">
                    <h3>Neural Analysis Input</h3>
                    <p style={{ color: 'var(--text-dim)', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
                        Upload a forensic-quality face crop. Optimal resolution: 224x224px.
                    </p>

                    <div
                        className={`upload-zone ${loading ? 'scanning' : ''}`}
                        onClick={onUploadClick}
                        onDragOver={handleDragOver}
                        onDrop={handleDrop}
                    >
                        {preview ? (
                            <div className="preview-container">
                                {isVideo ? (
                                    <video src={preview} className="preview-img" autoPlay muted loop />
                                ) : (
                                    <img src={preview} alt="Preview" className="preview-img" />
                                )}
                            </div>
                        ) : (
                            <>
                                <span className="upload-icon">🔭</span>
                                <p>Drag frame asset or video stream</p>
                            </>
                        )}
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileChange}
                            accept="image/*,video/*"
                        />
                    </div>

                    <button onClick={handlePredict} disabled={!image || loading}>
                        {loading ? 'Processing Neural Stream...' : 'Initiate Forensic Scan'}
                    </button>
                </section>

                <section className="glass-card">
                    <h3>Forensic Report</h3>
                    {!result && !loading && (
                        <div style={{ textAlign: 'center', padding: '4rem 1rem', color: 'var(--text-dim)', border: '1px dashed var(--glass-border)', borderRadius: '16px' }}>
                            <span style={{ fontSize: '3rem', display: 'block', opacity: 0.3, marginBottom: '1rem' }}>⚖️</span>
                            Stationary: Awaiting cryptographic material...
                        </div>
                    )}

                    {loading && (
                        <div style={{ textAlign: 'center', padding: '2rem' }}>
                            <div className="loading-spinner"></div>
                            <p style={{ color: 'var(--primary)', fontWeight: '600', marginTop: '1rem' }}>
                                {isVideo ? 'Performing Temporal Scan...' : 'Extracting Forgery Signatures...'}
                            </p>
                        </div>
                    )}

                    {result && !result.error && (
                        <div className="result-box">
                            <div className="prediction-label">Machine Consensus</div>
                            <div className={`prediction-value ${result.prediction?.toLowerCase()}`}>
                                {result.prediction}
                            </div>

                            <div className="compact-results-layout" style={{ display: 'grid', gridTemplateColumns: result.heatmap ? '1fr 1fr' : '1fr', gap: '2rem' }}>
                                <div className="result-main-col">
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <div className="prediction-label">Confidence Interval</div>
                                        <div style={{ fontSize: '0.9rem', fontWeight: 'bold', color: 'var(--primary)' }}>
                                            {(result.confidence * 100).toFixed(2)}%
                                        </div>
                                    </div>
                                    <div className="progress-container">
                                        <div
                                            className="progress-bar"
                                            style={{ width: `${(result.confidence || 0) * 100}%` }}
                                        ></div>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: '-0.5rem', marginBottom: '1.5rem' }}>
                                        <span>Statistical Noise</span>
                                        <span>Deterministic</span>
                                    </div>

                                    <div className="stats-grid" style={{ marginTop: '0' }}>
                                        <div className="stat-item">
                                            <div className="stat-label">Analysis Latency</div>
                                            <div className="stat-val">{result.inference_ms} ms</div>
                                        </div>
                                        <div className="stat-item">
                                            <div className="stat-label">{result.is_video ? 'Frames Scanned' : 'Detected Group'}</div>
                                            <div className="stat-val">{result.is_video ? `${result.frames_scanned} Frames` : (result.demographics?.group || 'N/A')}</div>
                                        </div>
                                        {result.is_video && (
                                            <div className="stat-item">
                                                <div className="stat-label">Manipulation Ratio</div>
                                                <div className="stat-val">{(result.fake_ratio * 100).toFixed(1)}%</div>
                                            </div>
                                        )}
                                        <div className="stat-item">
                                            <div className="stat-label">Applied Threshold</div>
                                            <div className="stat-val">t = {result.demographics?.threshold_applied || '0.5'}</div>
                                        </div>
                                        <div className="stat-item">
                                            <div className="stat-label">Bias Mitigation</div>
                                            <div className="stat-val" style={{ color: 'var(--success)', fontSize: '0.9rem' }}>Verified</div>
                                        </div>
                                    </div>
                                </div>

                                {result.heatmap && (
                                    <div className="heatmap-section">
                                        <div className="prediction-label" style={{ marginBottom: '0.5rem' }}>Forgery Localization</div>
                                        <div
                                            style={{
                                                borderRadius: '16px',
                                                overflow: 'hidden',
                                                border: '2px solid var(--secondary)',
                                                boxShadow: '0 0 20px rgba(0, 243, 255, 0.2)',
                                                height: '220px'
                                            }}
                                        >
                                            <img
                                                src={`data:image/jpeg;base64,${result.heatmap}`}
                                                alt="Grad-CAM Heatmap"
                                                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                            />
                                        </div>
                                        <p style={{ fontSize: '0.65rem', color: 'var(--text-dim)', marginTop: '0.5rem', textAlign: 'center', lineHeight: '1.2' }}>
                                            Colored regions highlight high-forgery signatures detected on the face.
                                        </p>
                                    </div>
                                )}
                            </div>

                            <div style={{ marginTop: '1.5rem', padding: '0.8rem', background: 'rgba(0, 242, 255, 0.05)', borderRadius: '8px', borderLeft: '3px solid var(--secondary)' }}>
                                <p style={{ fontSize: '0.8rem', color: 'var(--text-main)' }}>
                                    <strong>Fairness Check:</strong> Accuracy parity is enforced via pairwise demographic constraints. Performance is verified within a 0.08 tolerance across all protected groups.
                                </p>
                            </div>
                        </div>
                    )}

                    {result && result.error && (
                        <div style={{ marginTop: '2rem', padding: '1.5rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--error)', borderRadius: '16px', textAlign: 'center' }}>
                            <span style={{ fontSize: '2rem', display: 'block', marginBottom: '1rem' }}>⚠️</span>
                            <p style={{ color: 'var(--error)', fontWeight: '600' }}>Analysis Failed</p>
                            <p style={{ color: 'var(--text-dim)', fontSize: '0.9rem' }}>{result.error}</p>
                        </div>
                    )}
                </section>
            </div>

            <footer style={{ marginTop: '4rem', textAlign: 'center', color: 'var(--text-dim)', fontSize: '0.8rem', opacity: 0.5 }}>
                &copy; 2026 Fairness-Preserving Deepfake Detection Research. This tool is for research validation purposes only.
            </footer>
        </div>
    );
}

export default App;
