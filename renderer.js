class Renderer {
  constructor() {
    this.canvas = document.getElementById('gpuCanvas');
    this.ctx = this.canvas.getContext('webgpu');
    this.device = null;
    this.format = null;
    this.rotation = { x: 0, y: 0 };
    this.zoom = 3;
    this.vertices = null;
    this.frameCount = 0;
    this.lastTime = Date.now();
  }

  async init() {
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({
      device: gpuManager.device,
      format: this.format
    });

    this.device = gpuManager.device;

    // Create render pipeline
    const shaderCode = `
      struct VertexInput {
        @location(0) position: vec3f,
      }

      struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec4f,
      }

      @group(0) @binding(0) var<uniform> mvp: mat4x4f;

      @vertex
      fn vs(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        output.position = mvp * vec4f(input.position, 1.0);
        output.color = vec4f(0.2, 0.6, 1.0, 1.0);
        return output;
      }

      @fragment
      fn fs(input: VertexOutput) -> @location(0) vec4f {
        return input.color;
      }
    `;

    const shaderModule = this.device.createShaderModule({ code: shaderCode });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform' }
      }]
    });

    this.renderPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module: shaderModule,
        entryPoint: 'vs',
        buffers: [{
          arrayStride: 12,
          attributes: [{
            shaderLocation: 0,
            offset: 0,
            format: 'float32x3'
          }]
        }]
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs',
        targets: [{ format: this.format }]
      },
      primitive: { topology: 'triangle-list' }
    });

    this.mvpBuffer = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(this.mvpBuffer.getMappedRange()).fill(0);
    this.mvpBuffer.unmap();

    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.mvpBuffer } }]
    });

    // Setup controls
    this.setupControls();

    // Resize canvas
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
  }

  resizeCanvas() {
    this.canvas.width = this.canvas.clientWidth;
    this.canvas.height = this.canvas.clientHeight;
  }

  setupControls() {
    let isDragging = false;

    this.canvas.addEventListener('contextmenu', e => e.preventDefault());
    this.canvas.addEventListener('mousedown', e => {
      if (e.button === 2) isDragging = true;
    });
    this.canvas.addEventListener('mousemove', e => {
      if (isDragging) {
        this.rotation.y += e.movementX * 0.005;
        this.rotation.x += e.movementY * 0.005;
      }
    });
    this.canvas.addEventListener('mouseup', () => {
      isDragging = false;
    });
    this.canvas.addEventListener('wheel', e => {
      e.preventDefault();
      this.zoom += e.deltaY > 0 ? 0.1 : -0.1;
      this.zoom = Math.max(0.5, this.zoom);
    });
  }

  createMatrix() {
    const aspect = this.canvas.width / this.canvas.height;

    // Perspective projection
    const fov = Math.PI / 4;
    const f = 1 / Math.tan(fov / 2);
    const zNear = 0.1;
    const zFar = 100;

    const projection = new Float32Array(16);
    projection[0] = f / aspect;
    projection[5] = f;
    projection[10] = (zFar + zNear) / (zNear - zFar);
    projection[11] = -1;
    projection[14] = (2 * zFar * zNear) / (zNear - zFar);

    // View matrix (translate back)
    const view = new Float32Array(16);
    view[0] = view[5] = view[10] = view[15] = 1;
    view[14] = -this.zoom;

    // Model matrix (rotation + scale)
    const model = new Float32Array(16);
    model[0] = model[5] = model[10] = model[15] = 1;

    // Apply rotations
    const cx = Math.cos(this.rotation.x);
    const sx = Math.sin(this.rotation.x);
    const cy = Math.cos(this.rotation.y);
    const sy = Math.sin(this.rotation.y);

    // Simplified rotation - rotate around Y then X
    const m = new Float32Array(16);
    m[0] = cy;
    m[2] = sy;
    m[5] = 1;
    m[8] = -sy;
    m[10] = cy;
    m[15] = 1;

    const m2 = new Float32Array(16);
    m2[0] = 1;
    m2[5] = cx;
    m2[6] = -sx;
    m2[9] = sx;
    m2[10] = cx;
    m2[15] = 1;

    // Multiply matrices
    const mvp = this.multiplyMatrices(projection, this.multiplyMatrices(view, m2));
    return this.multiplyMatrices(mvp, model);
  }

  multiplyMatrices(a, b) {
    const result = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        let sum = 0;
        for (let k = 0; k < 4; k++) {
          sum += a[i * 4 + k] * b[k * 4 + j];
        }
        result[i * 4 + j] = sum;
      }
    }
    return result;
  }

  render() {
    const commandEncoder = this.device.createCommandEncoder();

    // Clear and setup render pass
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: this.ctx.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1 },
        loadOp: 'clear',
        storeOp: 'store'
      }]
    });

    if (this.vertices && this.indexBuffer && this.vertexBuffer) {
      const mvp = this.createMatrix();
      this.device.queue.writeBuffer(this.mvpBuffer, 0, mvp);

      renderPass.setPipeline(this.renderPipeline);
      renderPass.setBindGroup(0, this.bindGroup);
      renderPass.setVertexBuffer(0, this.vertexBuffer);
      renderPass.setIndexBuffer(this.indexBuffer, 'uint32');
      renderPass.drawIndexed(this.indexCount);
    }

    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);

    // FPS counter
    this.frameCount++;
    const now = Date.now();
    if (now - this.lastTime > 1000) {
      document.getElementById('fpsCounter').textContent = this.frameCount;
      this.frameCount = 0;
      this.lastTime = now;
    }
  }

  async updateVertices(vertices) {
    this.vertices = vertices;

    // Create vertex buffer
    if (!this.vertexBuffer) {
      this.vertexBuffer = this.device.createBuffer({
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
      });
      new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
      this.vertexBuffer.unmap();
    } else {
      this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices);
    }

    // Create index buffer (if not exists)
    if (!this.indexBuffer && stlOps.meshData) {
      const indices = stlOps.meshData.indices;
      this.indexCount = indices.length;
      this.indexBuffer = this.device.createBuffer({
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
      });
      new Uint32Array(this.indexBuffer.getMappedRange()).set(indices);
      this.indexBuffer.unmap();
    }
  }
}

// Initialize
const renderer = new Renderer();

document.addEventListener('DOMContentLoaded', async () => {
  try {
    // Initialize GPU
    const gpuInfo = await gpuManager.init();
    document.getElementById('gpuInfo').textContent = gpuInfo.info;

    // Initialize STL operations
    stlOps.generateCubeMesh();
    await stlOps.initComputePipeline();
    stlOps.createBuffers();
    stlOps.setupBindGroup(gpuManager.createBindGroupLayout([
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
    ]));

    // Initialize renderer
    await renderer.init();
    await renderer.updateVertices(stlOps.meshData.vertices);
    document.getElementById('vertexCount').textContent = stlOps.meshData.vertexCount;

    // Setup UI controls
    const operationSelect = document.getElementById('operationSelect');
    const paramSlider = document.getElementById('paramSlider');
    const paramValue = document.getElementById('paramValue');
    const processBtn = document.getElementById('processBtn');
    const resetBtn = document.getElementById('resetBtn');
    const computeTime = document.getElementById('computeTime');

    paramSlider.addEventListener('input', e => {
      paramValue.textContent = parseFloat(e.target.value).toFixed(1);
    });

    processBtn.addEventListener('click', async () => {
      processBtn.disabled = true;
      const operation = operationSelect.value;
      const param = parseFloat(paramSlider.value);

      const time = await stlOps.processOperation(operation, param);
      const vertices = await stlOps.getProcessedVertices();
      await renderer.updateVertices(vertices);

      computeTime.textContent = time.toFixed(2) + 'ms';
      processBtn.disabled = false;
    });

    resetBtn.addEventListener('click', async () => {
      stlOps.generateCubeMesh();
      stlOps.createBuffers();
      stlOps.setupBindGroup(gpuManager.createBindGroupLayout([
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
      ]));
      await renderer.updateVertices(stlOps.meshData.vertices);
      paramSlider.value = 1;
      paramValue.textContent = '1.0';
    });

    // Animation loop
    function animate() {
      renderer.render();
      requestAnimationFrame(animate);
    }
    animate();

  } catch (error) {
    console.error('Initialization failed:', error);
    document.body.innerHTML = `<div class="p-8 text-red-500"><h1>Error</h1><p>${error.message}</p></div>`;
  }
});