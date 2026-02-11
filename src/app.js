import { gpuManager } from './GpuUtils.js';
import { stlOps } from './StlOperations.js';
import { Renderer } from './Renderer.js';

async function initApp() {
  const canvas = document.getElementById('gpuCanvas');
  const gpuInfoEl = document.getElementById('gpuInfo');
  const vertexCountEl = document.getElementById('vertexCount');
  const computeTimeEl = document.getElementById('computeTime');
  const operationSelect = document.getElementById('operationSelect');
  const paramSlider = document.getElementById('paramSlider');
  const paramValueEl = document.getElementById('paramValue');
  const processBtn = document.getElementById('processBtn');
  const resetBtn = document.getElementById('resetBtn');
  const errorPanel = document.getElementById('errorPanel');

  try {
    const gpuInfo = await gpuManager.init();
    gpuInfoEl.textContent = gpuInfo.info;

    stlOps.generateCubeMesh();
    await stlOps.initComputePipeline();
    stlOps.createBuffers();
    stlOps.setupBindGroup();

    const renderer = new Renderer(canvas);
    await renderer.init();
    renderer.updateMesh(stlOps.meshData.vertices);
    vertexCountEl.textContent = stlOps.meshData.vertexCount;

    paramSlider.addEventListener('input', (e) => {
      paramValueEl.textContent = parseFloat(e.target.value).toFixed(1);
    });

    processBtn.addEventListener('click', async () => {
      processBtn.disabled = true;
      processBtn.textContent = 'Processing...';
      const operation = operationSelect.value;
      const param = parseFloat(paramSlider.value);

      const elapsed = await stlOps.processOperation(operation, param);
      const processed = await stlOps.getProcessedVertices();
      renderer.updateMesh(processed);

      computeTimeEl.textContent = elapsed.toFixed(2) + 'ms';
      processBtn.disabled = false;
      processBtn.textContent = 'Process';
    });

    resetBtn.addEventListener('click', () => {
      stlOps.generateCubeMesh();
      stlOps.createBuffers();
      stlOps.setupBindGroup();
      renderer.resetIndexBuffer();
      renderer.updateMesh(stlOps.meshData.vertices);
      vertexCountEl.textContent = stlOps.meshData.vertexCount;
      computeTimeEl.textContent = '0ms';
      paramSlider.value = 1;
      paramValueEl.textContent = '1.0';
    });

    function animate() {
      renderer.render();
      requestAnimationFrame(animate);
    }
    animate();
  } catch (err) {
    console.error('Init failed:', err);
    if (errorPanel) {
      errorPanel.classList.remove('hidden');
      errorPanel.querySelector('p').textContent = err.message;
    }
  }
}

document.addEventListener('DOMContentLoaded', initApp);
