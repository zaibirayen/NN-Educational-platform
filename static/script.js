const fileInput = document.getElementById("database");
const selectedFile = document.getElementById("selectedFile");
const trainForm = document.getElementById("trainForm");
const trainStatus = document.getElementById("trainStatus");

if (fileInput) {
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      selectedFile.textContent = fileInput.files[0].name;
    } else {
      selectedFile.textContent = "No file selected";
    }
  });
}

if (trainForm && trainStatus) {
  trainForm.addEventListener("submit", () => {
    trainStatus.textContent = "Training in progress… please wait.";
    trainStatus.style.display = "block";
  });
}

const architectureSelect = document.getElementById("model_architecture");
const hiddenLayersRow = document.getElementById("hidden_layers")?.closest(".form-row");
const neuronsRow = document.getElementById("neurons")?.closest(".form-row");
const activationRow = document.getElementById("activation")?.closest(".form-row");

function updateFormVisibility() {
  if (!architectureSelect) return;
  const selected = architectureSelect.value;
  const hideHidden = ["historical_perceptron", "single_perceptron", "sklearn_linear"].includes(selected);
  if (hiddenLayersRow) hiddenLayersRow.style.display = hideHidden ? "none" : "block";
  if (neuronsRow) neuronsRow.style.display = hideHidden ? "none" : "block";
  if (activationRow) activationRow.style.display = selected === "sklearn_linear" ? "none" : "block";
}

if (architectureSelect) {
  architectureSelect.addEventListener("change", updateFormVisibility);
  updateFormVisibility();
}

// Interactive prediction table functionality
const togglePredictions = document.getElementById("togglePredictions");
const toggleActual = document.getElementById("toggleActual");
const toggleErrors = document.getElementById("toggleErrors");
const predictionTable = document.getElementById("predictionTable");

if (togglePredictions && predictionTable) {
  togglePredictions.addEventListener("change", function() {
    const rows = predictionTable.querySelectorAll(".prediction-row");
    rows.forEach(row => {
      row.style.display = this.checked ? "" : "none";
    });
  });
}

if (toggleActual && predictionTable) {
  toggleActual.addEventListener("change", function() {
    const rows = predictionTable.querySelectorAll(".prediction-row");
    rows.forEach(row => {
      row.style.display = this.checked ? "" : "none";
    });
  });
}

if (toggleErrors && predictionTable) {
  toggleErrors.addEventListener("change", function() {
    const rows = predictionTable.querySelectorAll(".prediction-row");
    rows.forEach(row => {
      const isCorrect = row.dataset.correct === "True";
      if (this.checked) {
        row.classList.toggle("error-highlight", !isCorrect);
      } else {
        row.classList.remove("error-highlight");
      }
    });
  });
}

// 3D Network Visualization controls
const rotateLeft = document.getElementById("rotateLeft");
const rotateRight = document.getElementById("rotateRight");
const toggleView = document.getElementById("toggleView");
const network3D = document.getElementById("network3D");

let rotation = 0;
let is3D = true;

if (rotateLeft && network3D) {
  rotateLeft.addEventListener("click", function() {
    rotation -= 15;
    network3D.querySelector(".layer-visualization").style.transform = `rotateY(${rotation}deg)`;
  });
}

if (rotateRight && network3D) {
  rotateRight.addEventListener("click", function() {
    rotation += 15;
    network3D.querySelector(".layer-visualization").style.transform = `rotateY(${rotation}deg)`;
  });
}

if (toggleView && network3D) {
  toggleView.addEventListener("click", function() {
    is3D = !is3D;
    const neurons = network3D.querySelectorAll(".neuron-3d");
    neurons.forEach(neuron => {
      if (is3D) {
        neuron.querySelector(".neuron-sphere").style.borderRadius = "50%";
      } else {
        neuron.querySelector(".neuron-sphere").style.borderRadius = "2px";
      }
    });
  });
}
