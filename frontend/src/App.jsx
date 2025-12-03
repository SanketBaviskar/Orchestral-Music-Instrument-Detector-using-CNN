import { useState, useRef } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Music, X, Play, Pause, RefreshCw } from "lucide-react";
import "./App.css";

function App() {
	const [file, setFile] = useState(null);
	const [previewUrl, setPreviewUrl] = useState(null);
	const [loading, setLoading] = useState(false);
	const [result, setResult] = useState(null);
	const [error, setError] = useState(null);
	const [isPlaying, setIsPlaying] = useState(false);
	const audioRef = useRef(null);

	const handleFileChange = (e) => {
		const selectedFile = e.target.files[0];
		if (selectedFile) {
			setFile(selectedFile);
			setPreviewUrl(URL.createObjectURL(selectedFile));
			setResult(null);
			setError(null);
		}
	};

	const handleDrop = (e) => {
		e.preventDefault();
		const selectedFile = e.dataTransfer.files[0];
		if (selectedFile) {
			setFile(selectedFile);
			setPreviewUrl(URL.createObjectURL(selectedFile));
			setResult(null);
			setError(null);
		}
	};

	const handleDragOver = (e) => {
		e.preventDefault();
	};

	const handlePredict = async () => {
		if (!file) return;

		setLoading(true);
		setError(null);

		const formData = new FormData();
		formData.append("file", file);

		try {
			const response = await axios.post("/api/predict", formData, {
				headers: {
					"Content-Type": "multipart/form-data",
				},
			});
			setResult(response.data);
		} catch (err) {
			setError("Prediction failed. Please try again.");
			console.error(err);
		} finally {
			setLoading(false);
		}
	};

	const handleReset = () => {
		setFile(null);
		setPreviewUrl(null);
		setResult(null);
		setError(null);
		if (audioRef.current) {
			audioRef.current.pause();
			audioRef.current.currentTime = 0;
		}
		setIsPlaying(false);
	};

	const togglePlay = () => {
		if (audioRef.current) {
			if (isPlaying) {
				audioRef.current.pause();
			} else {
				audioRef.current.play();
			}
			setIsPlaying(!isPlaying);
		}
	};

	return (
		<div className="container">
			<header>
				<h1>Orchestral Instrument Detector</h1>
				<p>AI-powered instrument classification</p>
			</header>

			<main>
				<AnimatePresence mode="wait">
					{!file ? (
						<motion.div
							key="upload"
							initial={{ opacity: 0, y: 20 }}
							animate={{ opacity: 1, y: 0 }}
							exit={{ opacity: 0, y: -20 }}
							className="upload-area"
							onDrop={handleDrop}
							onDragOver={handleDragOver}
							onClick={() =>
								document.getElementById("fileInput").click()
							}
						>
							<Upload size={48} className="icon" />
							<p>Drag & Drop audio file here</p>
							<p className="small">or click to browse</p>
							<input
								type="file"
								id="fileInput"
								accept=".mp3, .wav"
								hidden
								onChange={handleFileChange}
							/>
						</motion.div>
					) : (
						<motion.div
							key="preview"
							initial={{ opacity: 0, y: 20 }}
							animate={{ opacity: 1, y: 0 }}
							exit={{ opacity: 0, y: -20 }}
							className="preview-area"
						>
							<div className="file-info">
								<Music size={24} />
								<span>{file.name}</span>
								<button
									onClick={handleReset}
									className="close-btn"
								>
									<X size={20} />
								</button>
							</div>

							<div className="audio-controls">
								<button
									onClick={togglePlay}
									className="play-btn"
								>
									{isPlaying ? (
										<Pause size={24} />
									) : (
										<Play size={24} />
									)}
								</button>
								<audio
									ref={audioRef}
									src={previewUrl}
									onEnded={() => setIsPlaying(false)}
								/>
							</div>

							{!result && !loading && (
								<button
									onClick={handlePredict}
									className="predict-btn"
								>
									Identify Instrument
								</button>
							)}

							{loading && (
								<div className="loading">
									<RefreshCw size={32} className="spinner" />
									<p>Analyzing spectrogram...</p>
								</div>
							)}

							{result && (
								<motion.div
									initial={{ opacity: 0, scale: 0.9 }}
									animate={{ opacity: 1, scale: 1 }}
									className="result-card"
								>
									<h2>{result.label}</h2>
									<div className="confidence-bar-bg">
										<motion.div
											initial={{ width: 0 }}
											animate={{
												width: `${
													result.confidence * 100
												}%`,
											}}
											transition={{
												duration: 1,
												ease: "easeOut",
											}}
											className="confidence-bar-fill"
										/>
									</div>
									<p>
										{Math.round(result.confidence * 100)}%
										Confidence
									</p>
								</motion.div>
							)}

							{error && <p className="error">{error}</p>}
						</motion.div>
					)}
				</AnimatePresence>
			</main>
		</div>
	);
}

export default App;
