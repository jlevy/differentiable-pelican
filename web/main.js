import init, {
  PelicanOptimizer,
  render_initial_pelican,
  initial_pelican_svg,
} from "./pkg/pelican_wasm.js";

let optimizer = null;
let animationId = null;

const targetCanvas = document.getElementById("target-canvas");
const renderCanvas = document.getElementById("render-canvas");
const targetCtx = targetCanvas.getContext("2d");
const renderCtx = renderCanvas.getContext("2d");
const statusEl = document.getElementById("status");
const lossEl = document.getElementById("loss-display");
const svgContainer = document.getElementById("svg-container");
const btnStart = document.getElementById("btn-start");
const btnReset = document.getElementById("btn-reset");

function getParams() {
  return {
    resolution: parseInt(document.getElementById("resolution").value, 10),
    steps: parseInt(document.getElementById("steps").value, 10),
    batch: parseInt(document.getElementById("batch").value, 10),
  };
}

// Draw grayscale u8 pixels to a canvas (resize canvas if needed)
function drawGrayscale(canvas, ctx, pixels, width, height) {
  canvas.width = width;
  canvas.height = height;
  const imageData = ctx.createImageData(width, height);
  for (let i = 0; i < pixels.length; i++) {
    const v = pixels[i];
    imageData.data[i * 4 + 0] = v;
    imageData.data[i * 4 + 1] = v;
    imageData.data[i * 4 + 2] = v;
    imageData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

// Render initial pelican on both canvases
function showInitialPelican(resolution) {
  const pixels = render_initial_pelican(resolution, resolution);
  drawGrayscale(targetCanvas, targetCtx, pixels, resolution, resolution);
  drawGrayscale(renderCanvas, renderCtx, pixels, resolution, resolution);

  const svg = initial_pelican_svg(256, 256);
  svgContainer.innerHTML = svg;
}

// Get target canvas pixels as grayscale u8 array
function getTargetPixels(resolution) {
  // Draw the target canvas content at the target resolution
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = resolution;
  tempCanvas.height = resolution;
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(targetCanvas, 0, 0, resolution, resolution);
  const imageData = tempCtx.getImageData(0, 0, resolution, resolution);
  const pixels = new Uint8Array(resolution * resolution);
  for (let i = 0; i < pixels.length; i++) {
    // Convert RGBA to grayscale (simple average)
    const r = imageData.data[i * 4];
    const g = imageData.data[i * 4 + 1];
    const b = imageData.data[i * 4 + 2];
    pixels[i] = Math.round((r + g + b) / 3);
  }
  return pixels;
}

function stopOptimization() {
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
}

function startOptimization() {
  stopOptimization();

  const { resolution, steps, batch } = getParams();
  const targetPixels = getTargetPixels(resolution);

  if (optimizer) {
    optimizer.free();
    optimizer = null;
  }

  statusEl.textContent = `Initializing optimizer (${resolution}x${resolution}, ${steps} steps)...`;
  btnStart.disabled = true;

  // Use setTimeout to let the UI update before the potentially slow constructor
  setTimeout(() => {
    try {
      optimizer = new PelicanOptimizer(targetPixels, resolution, resolution, steps);
    } catch (e) {
      statusEl.textContent = `Error: ${e}`;
      btnStart.disabled = false;
      return;
    }

    const losses = [];

    function optimizationLoop() {
      if (!optimizer || optimizer.is_done()) {
        statusEl.textContent = `Done! ${losses.length} steps completed.`;
        btnStart.disabled = false;
        updateDisplay(resolution);
        return;
      }

      const batchLosses = optimizer.step_n(batch);
      for (const l of batchLosses) {
        losses.push(l);
      }

      const step = optimizer.current_step();
      const lastLoss = losses[losses.length - 1];
      statusEl.textContent = `Step ${step}/${steps} | Loss: ${lastLoss.toFixed(6)}`;
      lossEl.textContent = `Loss history: ${losses.slice(-10).map(l => l.toFixed(4)).join(" → ")}`;

      updateDisplay(resolution);
      animationId = requestAnimationFrame(optimizationLoop);
    }

    animationId = requestAnimationFrame(optimizationLoop);
  }, 50);
}

function updateDisplay(resolution) {
  if (!optimizer) return;

  const pixels = optimizer.get_rendered_pixels(resolution, resolution);
  drawGrayscale(renderCanvas, renderCtx, pixels, resolution, resolution);

  const svg = optimizer.get_svg(256, 256);
  svgContainer.innerHTML = svg;
}

function reset() {
  stopOptimization();
  if (optimizer) {
    optimizer.free();
    optimizer = null;
  }
  const { resolution } = getParams();
  showInitialPelican(resolution);
  statusEl.textContent = "Reset. Ready to optimize.";
  lossEl.textContent = "";
  btnStart.disabled = false;
}

async function main() {
  try {
    await init();
  } catch (e) {
    document.getElementById("loading").textContent = `Failed to load WASM: ${e}`;
    return;
  }

  document.getElementById("loading").style.display = "none";
  document.getElementById("app").style.display = "block";

  const { resolution } = getParams();
  showInitialPelican(resolution);

  btnStart.addEventListener("click", startOptimization);
  btnReset.addEventListener("click", reset);

  // Drawing on target canvas
  let drawing = false;
  const getCursorPos = (e) => {
    const rect = targetCanvas.getBoundingClientRect();
    const scaleX = targetCanvas.width / rect.width;
    const scaleY = targetCanvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  targetCanvas.addEventListener("mousedown", (e) => {
    drawing = true;
    const pos = getCursorPos(e);
    targetCtx.fillStyle = "black";
    targetCtx.beginPath();
    targetCtx.arc(pos.x, pos.y, 3, 0, Math.PI * 2);
    targetCtx.fill();
  });

  targetCanvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    const pos = getCursorPos(e);
    targetCtx.fillStyle = "black";
    targetCtx.beginPath();
    targetCtx.arc(pos.x, pos.y, 3, 0, Math.PI * 2);
    targetCtx.fill();
  });

  targetCanvas.addEventListener("mouseup", () => { drawing = false; });
  targetCanvas.addEventListener("mouseleave", () => { drawing = false; });
}

main();
