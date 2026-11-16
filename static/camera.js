let cameraStream = null;

async function startCamera() {
  const video = document.getElementById("camera");
  const status = document.getElementById("cameraStatus");
  if (!video) return;
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Camera access requires HTTPS or localhost and a supported browser.");
    }
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    video.srcObject = cameraStream;
    await video.play();
    if (status) status.textContent = "Camera is ready.";
  } catch (err) {
    if (status) status.textContent = "Camera error: " + err.message;
  }
}

function captureDataUrl() {
  const video = document.getElementById("camera");
  const canvas = document.getElementById("captureCanvas");
  if (!video || !canvas || !video.videoWidth) {
    throw new Error("Start the camera first.");
  }
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.88);
}

function drawBoundingBoxes(boxes) {
  const video = document.getElementById("camera");
  const overlay = document.getElementById("overlayCanvas");
  if (!video || !overlay) return;

  overlay.width = video.clientWidth;
  overlay.height = video.clientHeight;
  const scaleX = overlay.width / video.videoWidth;
  const scaleY = overlay.height / video.videoHeight;

  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.font = "bold 14px Arial";

  (boxes || []).forEach(item => {
    const [x, y, w, h] = item.bbox;
    const sx = x * scaleX;
    const sy = y * scaleY;
    const sw = w * scaleX;
    const sh = h * scaleY;

    ctx.strokeStyle = item.matched ? "#22c55e" : "#f59e0b";
    ctx.lineWidth = 3;
    ctx.strokeRect(sx, sy, sw, sh);

    ctx.fillStyle = item.matched ? "#16a34a" : "#d97706";
    const textWidth = ctx.measureText(item.label).width + 10;
    ctx.fillRect(sx, Math.max(0, sy - 22), textWidth, 22);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(item.label, sx + 5, Math.max(15, sy - 6));
  });
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-camera-start]").forEach(btn => {
    btn.addEventListener("click", startCamera);
  });

  document.querySelectorAll("[data-camera-capture]").forEach(btn => {
    btn.addEventListener("click", () => {
      const status = document.getElementById("cameraStatus");
      try {
        document.getElementById("faceImage").value = captureDataUrl();
        if (status) status.textContent = "Face image captured. Click Save Student.";
      } catch (err) {
        if (status) status.textContent = err.message;
      }
    });
  });

  const recognizeButton = document.getElementById("recognizeButton");
  if (recognizeButton) {
    recognizeButton.addEventListener("click", async () => {
      const status = document.getElementById("cameraStatus");
      const result = document.getElementById("recognitionResult");
      recognizeButton.disabled = true;
      try {
        const image = captureDataUrl();
        status.textContent = "Scanning all faces in frame...";
        const response = await fetch(recognizeButton.dataset.recognizeUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image })
        });
        const data = await response.json();
        drawBoundingBoxes(data.boxes);
        result.textContent = data.message;
        result.className = "result " + (data.ok ? "success" : "warning");
        status.textContent = data.ok ? "Multi-face scan complete. Refreshing roster..." : "Try adjusting lighting or camera angle.";
        if (data.ok) {
          setTimeout(() => window.location.reload(), 1400);
        }
      } catch (err) {
        result.textContent = err.message;
        result.className = "result warning";
      } finally {
        recognizeButton.disabled = false;
      }
    });
  }
});
