const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const previewArea = document.getElementById("previewArea");
const fileName = document.getElementById("fileName");
const audioPlayer = document.getElementById("audioPlayer");
const predictBtn = document.getElementById("predictBtn");
const resetBtn = document.getElementById("resetBtn");
const loading = document.getElementById("loading");
const resultArea = document.getElementById("resultArea");
const instrumentName = document.getElementById("instrumentName");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceValue = document.getElementById("confidenceValue");

let selectedFile = null;

// Drag & Drop
uploadArea.addEventListener("dragover", (e) => {
	e.preventDefault();
	uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
	uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
	e.preventDefault();
	uploadArea.classList.remove("dragover");
	const file = e.dataTransfer.files[0];
	handleFile(file);
});

uploadArea.addEventListener("click", () => {
	fileInput.click();
});

fileInput.addEventListener("change", (e) => {
	const file = e.target.files[0];
	handleFile(file);
});

function handleFile(file) {
	if (
		file &&
		(file.type === "audio/mpeg" ||
			file.type === "audio/wav" ||
			file.name.endsWith(".mp3"))
	) {
		selectedFile = file;
		fileName.textContent = file.name;
		audioPlayer.src = URL.createObjectURL(file);

		uploadArea.style.display = "none";
		previewArea.style.display = "block";
		resultArea.style.display = "none";
	} else {
		alert("Please upload a valid MP3 or WAV file.");
	}
}

resetBtn.addEventListener("click", () => {
	selectedFile = null;
	fileInput.value = "";
	uploadArea.style.display = "block";
	previewArea.style.display = "none";
	resultArea.style.display = "none";
	loading.style.display = "none";
});

predictBtn.addEventListener("click", async () => {
	if (!selectedFile) return;

	loading.style.display = "block";
	resultArea.style.display = "none";
	predictBtn.disabled = true;

	const formData = new FormData();
	formData.append("file", selectedFile);

	try {
		const response = await fetch("/predict", {
			method: "POST",
			body: formData,
		});

		if (!response.ok) {
			throw new Error("Prediction failed");
		}

		const data = await response.json();
		showResult(data);
	} catch (error) {
		alert("Error: " + error.message);
	} finally {
		loading.style.display = "none";
		predictBtn.disabled = false;
	}
});

function showResult(data) {
	resultArea.style.display = "block";
	instrumentName.textContent = data.label;

	// Animate confidence bar
	const percentage = Math.round(data.confidence * 100);
	confidenceValue.textContent = percentage + "%";

	// Reset width first to trigger animation
	confidenceBar.style.width = "0%";
	setTimeout(() => {
		confidenceBar.style.width = percentage + "%";
	}, 100);
}
