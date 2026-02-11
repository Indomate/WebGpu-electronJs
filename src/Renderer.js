import { gpuManager } from './GpuUtils.js';
import { stlOps } from './StlOperations.js';

const RENDER_SHADER = `
  struct VertexOutput {
    @builtin(position) clip_pos: vec4f,
    @location(0) world_pos: vec3f,
  }

  @group(0) @binding(0) var<uniform> mvp: mat4x4f;

  @vertex
  fn vs(@location(0) position: vec3f) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = mvp * vec4f(position, 1.0);
    out.world_pos = position;
    return out;
  }

  @fragment
  fn fs(input: VertexOutput) -> @location(0) vec4f {
    let dx = dpdx(input.world_pos);
    let dy = dpdy(input.world_pos);
    let normal = normalize(cross(dx, dy));
    let light_dir = normalize(vec3f(0.6, 0.8, 0.5));
    let diffuse = max(dot(normal, light_dir), 0.0);
    let ambient = 0.18;
    let base_color = vec3f(0.25, 0.65, 0.9);
    let color = base_color * (ambient + diffuse * 0.82);
    return vec4f(color, 1.0);
  }
`;

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = null;
    this.device = null;
    this.format = null;
    this.rotation = { x: -0.4, y: 0.6 };
    this.zoom = 2.5;
    this.vertexBuffer = null;
    this.indexBuffer = null;
    this.indexCount = 0;
    this.depthTexture = null;
    this.frameCount = 0;
    this.lastFpsTime = performance.now();
  }

  async init() {
    this.device = gpuManager.device;
    this.format = navigator.gpu.getPreferredCanvasFormat();

    this.ctx = this.canvas.getContext('webgpu');
    this.ctx.configure({ device: this.device, format: this.format });

    const shaderModule = this.device.createShaderModule({ code: RENDER_SHADER });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform' },
      }],
    });

    this.renderPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module: shaderModule,
        entryPoint: 'vs',
        buffers: [{
          arrayStride: 12,
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
        }],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs',
        targets: [{ format: this.format }],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    this.mvpBuffer = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.mvpBuffer } }],
    });

    this.setupControls();
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
  }

  resizeCanvas() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;

    this.canvas.width = w;
    this.canvas.height = h;

    if (this.depthTexture) this.depthTexture.destroy();
    this.depthTexture = this.device.createTexture({
      size: [w, h],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  setupControls() {
    let dragging = false;

    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    this.canvas.addEventListener('mousedown', (e) => {
      if (e.button === 0 || e.button === 2) dragging = true;
    });
    window.addEventListener('mouseup', () => { dragging = false; });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      this.rotation.y += e.movementX * 0.005;
      this.rotation.x += e.movementY * 0.005;
    });
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.zoom *= e.deltaY > 0 ? 1.08 : 0.92;
      this.zoom = Math.max(0.5, Math.min(10, this.zoom));
    }, { passive: false });
  }

  buildMVP() {
    const aspect = this.canvas.width / this.canvas.height;
    const fov = Math.PI / 4;
    const f = 1 / Math.tan(fov / 2);
    const near = 0.1;
    const far = 100;

    const proj = new Float32Array(16);
    proj[0] = f / aspect;
    proj[5] = f;
    proj[10] = (far + near) / (near - far);
    proj[11] = -1;
    proj[14] = (2 * far * near) / (near - far);

    const view = new Float32Array(16);
    view[0] = 1; view[5] = 1; view[10] = 1; view[15] = 1;
    view[14] = -this.zoom;

    const cx = Math.cos(this.rotation.x), sx = Math.sin(this.rotation.x);
    const cy = Math.cos(this.rotation.y), sy = Math.sin(this.rotation.y);

    const rotY = new Float32Array(16);
    rotY[0] = cy;  rotY[2] = sy;
    rotY[5] = 1;
    rotY[8] = -sy; rotY[10] = cy;
    rotY[15] = 1;

    const rotX = new Float32Array(16);
    rotX[0] = 1;
    rotX[5] = cx;  rotX[6] = -sx;
    rotX[9] = sx;  rotX[10] = cx;
    rotX[15] = 1;

    const model = this.mat4Mul(rotX, rotY);
    const viewModel = this.mat4Mul(view, model);
    return this.mat4Mul(proj, viewModel);
  }

  mat4Mul(a, b) {
    const r = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        r[i * 4 + j] =
          a[i * 4 + 0] * b[0 * 4 + j] +
          a[i * 4 + 1] * b[1 * 4 + j] +
          a[i * 4 + 2] * b[2 * 4 + j] +
          a[i * 4 + 3] * b[3 * 4 + j];
      }
    }
    return r;
  }

  updateMesh(vertices) {
    if (this.vertexBuffer) this.vertexBuffer.destroy();
    this.vertexBuffer = this.device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
    this.vertexBuffer.unmap();

    if (!this.indexBuffer && stlOps.meshData) {
      const idx = stlOps.meshData.indices;
      this.indexCount = idx.length;
      this.indexBuffer = this.device.createBuffer({
        size: idx.byteLength,
        usage: GPUBufferUsage.INDEX,
        mappedAtCreation: true,
      });
      new Uint32Array(this.indexBuffer.getMappedRange()).set(idx);
      this.indexBuffer.unmap();
    }
  }

  resetIndexBuffer() {
    if (this.indexBuffer) {
      this.indexBuffer.destroy();
      this.indexBuffer = null;
    }
  }

  render() {
    if (!this.depthTexture || this.canvas.width === 0) return;

    const mvp = this.buildMVP();
    this.device.queue.writeBuffer(this.mvpBuffer, 0, mvp);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.ctx.getCurrentTexture().createView(),
        clearValue: { r: 0.08, g: 0.08, b: 0.12, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    if (this.vertexBuffer && this.indexBuffer) {
      pass.setPipeline(this.renderPipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.setVertexBuffer(0, this.vertexBuffer);
      pass.setIndexBuffer(this.indexBuffer, 'uint32');
      pass.drawIndexed(this.indexCount);
    }

    pass.end();
    this.device.queue.submit([encoder.finish()]);

    this.frameCount++;
    const now = performance.now();
    if (now - this.lastFpsTime >= 1000) {
      const fpsEl = document.getElementById('fpsCounter');
      if (fpsEl) fpsEl.textContent = this.frameCount;
      this.frameCount = 0;
      this.lastFpsTime = now;
    }
  }
}
