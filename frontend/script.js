const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const result = document.getElementById("result");

// Settings elements
const useContextCheckbox = document.getElementById("useContext");
const datasetSelect = document.getElementById("datasetSelect");

// Map dataset names to model paths
const datasetToModelPath = {
  "emnist_byclass": "models/cnn_emnist_byclass.pth",
  "emnist_letters": "models/cnn_emnist_letters.pth",
  "mnist": "models/baseline_mnist.pth"
};

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

function getSettings() {
  const dataset = datasetSelect.value;
  return {
    use_context: useContextCheckbox.checked.toString().toLowerCase(),
    dataset: dataset,
    model_path: datasetToModelPath[dataset]
  };
}

async function predictImage() {
  const file = imageInput.files[0];

  if (!file) {
    result.textContent = "Please upload an image first.";
    return;
  }

  const formData = new FormData();
  formData.append("image", file);
  
  const settings = getSettings();
  formData.append("use_context", settings.use_context);
  formData.append("dataset", settings.dataset);
  formData.append("model_path", settings.model_path);

  result.textContent = "Predicting...";

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (data.error) {
      result.textContent = data.error;
    } else {
      result.textContent = `Prediction: ${data.prediction}`;
    }
    result.scrollIntoView({ behavior: 'smooth' });
  } catch (error) {
    result.textContent = "Error connecting to backend.";
    result.scrollIntoView({ behavior: 'smooth' });
  }
}

// Drawing canvas code

const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");

canvas.style.touchAction = "none";

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.strokeStyle = "black";
ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.lineJoin = "round";

let drawing = false;

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);

function getCanvasPosition(e) {
  const rect = canvas.getBoundingClientRect();

  return {
    x: (e.clientX - rect.left) * (canvas.width / rect.width),
    y: (e.clientY - rect.top) * (canvas.height / rect.height)
  };
}

function startDrawing(e) {
  e.preventDefault();

  drawing = true;
  canvas.setPointerCapture(e.pointerId);

  const position = getCanvasPosition(e);

  ctx.beginPath();
  ctx.moveTo(position.x, position.y);
}

function draw(e) {
  e.preventDefault();

  if (!drawing) return;

  const position = getCanvasPosition(e);

  ctx.lineTo(position.x, position.y);
  ctx.stroke();
}

function stopDrawing(e) {
  e.preventDefault();

  drawing = false;

  if (canvas.hasPointerCapture(e.pointerId)) {
    canvas.releasePointerCapture(e.pointerId);
  }
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
    
    const settings = getSettings();
    formData.append("use_context", settings.use_context);
    formData.append("dataset", settings.dataset);
    formData.append("model_path", settings.model_path);

    result.textContent = "Predicting...";

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (data.error) {
        result.textContent = data.error;
      } else {
        result.textContent = `Prediction: ${data.prediction}`;
      }
      result.scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
      result.textContent = "Error connecting to backend.";
      result.scrollIntoView({ behavior: 'smooth' });
    }
  }, "image/png");
}