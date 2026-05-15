const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const result = document.getElementById("result");

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];

  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
    const placeholder = document.querySelector(".preview-placeholder");
    if (placeholder) placeholder.style.display = "none";
    result.textContent = "";
  }
});

async function predictImage() {
  const file = imageInput.files[0];

  if (!file) {
    result.textContent = "Please upload an image first.";
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  result.textContent = "Predicting...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (data.error) {
      result.textContent = data.error;
    } else {
      result.textContent = `Prediction: ${data.prediction}`;
    }
  } catch (error) {
    result.textContent = "Error connecting to backend.";
  }
}

// Drawing canvas code

const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.strokeStyle = "black";
ctx.lineWidth = 20;
ctx.lineCap = "round";

let drawing = false;

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);

function startDrawing(e) {
  drawing = true;
  const rect = canvas.getBoundingClientRect();
  ctx.beginPath();
  ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

function draw(e) {
  if (!drawing) return;

  const rect = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.stroke();
}

function stopDrawing() {
  drawing = false;
}

function clearCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  result.textContent = "";
}

async function predictDrawing() {
  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("image", blob, "drawing.png");

    result.textContent = "Predicting...";

    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    result.textContent = `Prediction: ${data.prediction}`;
  }, "image/png");
}