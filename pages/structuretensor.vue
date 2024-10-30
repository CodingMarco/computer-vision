<template>
  <div>
    <NuxtLayout>
      <NuxtPage />
      <h1 class="text-xl font-bold mb-5">Structure Tensor</h1>
      <div class="flex flex-row">
        <div class="flex flex-col">
          <div class="grid grid-cols-4 items-center gap-2 h-min">
            <label for="kernelSizeRange" class="font-medium"
              >Kernel Size:</label
            >
            <input
              id="kernelSizeRange"
              type="range"
              min="1"
              max="51"
              step="2"
              v-model="kernelSize"
              class="w-full col-span-2 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
            />
            <span class="ml-2">{{ kernelSize }}</span>

            <label for="sigmaRange" class="font-medium">Sigma:</label>
            <input
              id="sigmaRange"
              type="range"
              min="0.1"
              max="30"
              step="0.1"
              v-model="kernelSigma"
              class="w-full h-2 col-span-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
            />
            <span class="ml-2">{{ kernelSigma }}</span>
            <div class="h-5 col-span-4"></div>
            <span class="font-bold">λ<sub>1</sub> / 1000 =</span>
            <span class="col-span-3">{{ lambda1 }}</span>
            <span class="font-bold">λ<sub>2</sub> / 1000 =</span>
            <span class="col-span-3">{{ lambda2 }}</span>
          </div>
        </div>
        <canvas ref="canvas" width="644" height="644"></canvas>
      </div>
    </NuxtLayout>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from "vue";
import { matrix, eigs } from "mathjs";

const canvas = ref<HTMLCanvasElement>(null as any);

const ctx = computed(
  () => canvas.value?.getContext("2d") as CanvasRenderingContext2D
);
const kernelSize = ref(15);
const kernelSigma = ref(3);
const gaussianKernel = computed(() =>
  generateGaussianKernel(kernelSize.value, kernelSigma.value)
);
const lambda1 = ref("0");
const lambda2 = ref("0");
let image: HTMLImageElement;
let gradients: { Ix: number[]; Iy: number[] };

// Load image
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = src;
  });
}

function generateGaussianKernel(size: number, sigma: number) {
  if (size % 2 === 0) {
    throw new Error("Size must be an odd integer.");
  }

  const kernel = [];
  const mean = Math.floor(size / 2);
  const sigma2 = sigma * sigma;
  let sum = 0;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - mean;
      const dy = y - mean;
      const value =
        (1 / (2 * Math.PI * sigma2)) *
        Math.exp(-(dx * dx + dy * dy) / (2 * sigma2));
      kernel.push(value);
      sum += value;
    }
  }

  // Normalize the kernel so that the sum is 1
  return kernel.map((value) => value / sum);
}

// Compute image gradients Ix and Iy
function computeGradients() {
  const width = canvas.value.width;
  const height = canvas.value.height;
  const imageData = ctx.value.getImageData(0, 0, width, height);
  const data = imageData.data; // Image data in RGBA format

  // Initialize gradient arrays
  const Ix = new Array(width * height).fill(0);
  const Iy = new Array(width * height).fill(0);

  // Sobel kernels for x and y gradients
  const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

  // Function to get grayscale value at (x, y)
  function getGray(x: number, y: number) {
    const idx = (y * width + x) * 4; // RGBA index
    const r = data[idx];
    const g = data[idx + 1];
    const b = data[idx + 2];
    return 0.3 * r + 0.59 * g + 0.11 * b; // Standard grayscale conversion
  }

  // Convolve the Sobel kernels over each pixel (excluding borders)
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0,
        gy = 0;

      // Apply the Sobel kernels
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const pixelGray = getGray(x + kx, y + ky);
          const kernelIndex = (ky + 1) * 3 + (kx + 1);
          gx += pixelGray * sobelX[kernelIndex];
          gy += pixelGray * sobelY[kernelIndex];
        }
      }

      // Store the gradients at (x, y)
      const index = y * width + x;
      Ix[index] = gx;
      Iy[index] = gy;
    }
  }

  gradients = { Ix, Iy }; // Store gradients for access later
}

// Compute structure tensor and eigenvectors at given (x, y)
function computeEigenvectors(x: number, y: number) {
  const width = canvas.value.width;
  const height = canvas.value.height;

  const halfKernel = Math.floor(kernelSize.value / 2);

  // Helper to apply Gaussian convolution to a gradient component
  function applyGaussian(x: number, y: number, componentArray: number[]) {
    let sum = 0;
    for (let ky = -halfKernel; ky <= halfKernel; ky++) {
      for (let kx = -halfKernel; kx <= halfKernel; kx++) {
        const pixelX = Math.min(Math.max(x + kx, 0), width - 1);
        const pixelY = Math.min(Math.max(y + ky, 0), height - 1);
        const kernelIndex =
          (ky + halfKernel) * kernelSize.value + (kx + halfKernel);
        const pixelIndex = pixelY * width + pixelX;
        sum += componentArray[pixelIndex] * gaussianKernel.value[kernelIndex];
      }
    }
    return sum;
  }

  // Apply Gaussian smoothing on structure tensor components
  const Ixx = applyGaussian(
    x,
    y,
    gradients.Ix.map((v) => v * v)
  );
  const Iyy = applyGaussian(
    x,
    y,
    gradients.Iy.map((v) => v * v)
  );
  const Ixy = applyGaussian(
    x,
    y,
    gradients.Ix.map((v, i) => v * gradients.Iy[i])
  );

  // Construct the smoothed structure tensor at (x, y)
  const M = matrix([
    [Ixx, Ixy],
    [Ixy, Iyy],
  ]);

  // Compute eigenvalues and eigenvectors of the structure tensor
  return eigs(M).eigenvectors;
}

// Draw on canvas based on cursor position
function drawOverlay(x: number, y: number) {
  ctx.value.clearRect(0, 0, canvas.value.width, canvas.value.height);
  ctx.value.drawImage(image, 0, 0); // Redraw the base image

  const eigenvectors = computeEigenvectors(x, y);

  //const eigen1 = eigenvectors[0].value as number;
  //const eigen2 = eigenvectors[1].value as number;
  //lambda1.value = eigen1 > eigen2 ? eigen1.toFixed() : eigen2.toFixed();
  //lambda2.value = eigen1 > eigen2 ? eigen2.toFixed() : eigen1.toFixed();

  // Normalize eigenvalues
  const eigenvalues = eigenvectors.map((vec) => (vec.value as number) / 1000);
  lambda1.value = eigenvalues.sort()[1].toFixed();
  lambda2.value = eigenvalues.sort()[0].toFixed();

  // Draw gaussian kernel at cursor position
  const halfKernel = Math.floor(kernelSize.value / 2);
  const kernelMax = Math.max(...gaussianKernel.value);
  for (let ky = -halfKernel; ky <= halfKernel; ky++) {
    for (let kx = -halfKernel; kx <= halfKernel; kx++) {
      const pixelX = Math.min(Math.max(x + kx, 0), canvas.value.width - 1);
      const pixelY = Math.min(Math.max(y + ky, 0), canvas.value.height - 1);
      const kernelIndex =
        (ky + halfKernel) * kernelSize.value + (kx + halfKernel);
      const value = gaussianKernel.value[kernelIndex] / kernelMax;
      ctx.value.fillStyle = `rgba(0, 0, 0, ${value})`;
      ctx.value.fillRect(pixelX, pixelY, 1, 1);
    }
  }

  // Draw eigenvectors as arrows
  eigenvectors.forEach((vec, i) => {
    ctx.value.beginPath();
    ctx.value.moveTo(x, y);
    ctx.value.lineTo(
      x + vec.vector.get([0]) * eigenvalues[i],
      y + vec.vector.get([1]) * eigenvalues[i]
    );
    ctx.value.stroke();
  });
}

// Main function to set up canvas and interactivity
async function init() {
  image = await loadImage("/computer-vision/house_drawing.png");
  ctx.value.drawImage(image, 0, 0, canvas.value.width, canvas.value.height);
  computeGradients();

  // Add mousemove event listener
  canvas.value.addEventListener("mousemove", (e) => {
    const rect = canvas.value.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);
    drawOverlay(x, y);
  });
}

onMounted(init);
</script>
