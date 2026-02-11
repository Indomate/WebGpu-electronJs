import { gpuManager } from './GpuUtils.js';

const COMPUTE_SHADER = `
  struct Params {
    operation: u32,
    param_value: f32,
    vertex_count: u32,
    _pad: u32,
  }

  @group(0) @binding(0) var<storage, read> input_verts: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output_verts: array<f32>;
  @group(0) @binding(2) var<uniform> params: Params;

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.vertex_count) {
      return;
    }

    let base = idx * 3u;
    let x = input_verts[base];
    let y = input_verts[base + 1u];
    let z = input_verts[base + 2u];

    var ox = x;
    var oy = y;
    var oz = z;

    if (params.operation == 0u) {
      ox = x * params.param_value;
      oy = y * params.param_value;
      oz = z * params.param_value;
    } else if (params.operation == 1u) {
      let angle = params.param_value * 3.14159265;
      let c = cos(angle);
      let s = sin(angle);
      ox = x * c + z * s;
      oz = -x * s + z * c;
    } else if (params.operation == 2u) {
      oy = y + (params.param_value - 1.0) * 0.5;
    } else if (params.operation == 3u) {
      let len = sqrt(x * x + y * y + z * z);
      if (len > 0.001) {
        let nx = x / len;
        let ny = y / len;
        let nz = z / len;
        let r = 0.5;
        let t = clamp(params.param_value, 0.0, 2.0) - 1.0;
        ox = x + (nx * r - x) * t;
        oy = y + (ny * r - y) * t;
        oz = z + (nz * r - z) * t;
      }
    }

    output_verts[base] = ox;
    output_verts[base + 1u] = oy;
    output_verts[base + 2u] = oz;
  }
`;

const OP_MAP = { scale: 0, rotate: 1, translate: 2, inflate: 3 };

export class StlOperations {
  constructor() {
    this.meshData = null;
    this.computePipeline = null;
    this.inputBuffer = null;
    this.outputBuffer = null;
    this.uniformBuffer = null;
    this.bindGroup = null;
  }

  generateCubeMesh() {
    const positions = new Float32Array([
      -0.5, -0.5, -0.5,
       0.5, -0.5, -0.5,
       0.5,  0.5, -0.5,
      -0.5,  0.5, -0.5,
      -0.5, -0.5,  0.5,
       0.5, -0.5,  0.5,
       0.5,  0.5,  0.5,
      -0.5,  0.5,  0.5,
    ]);

    const indices = new Uint32Array([
      4, 5, 6, 4, 6, 7,
      1, 0, 3, 1, 3, 2,
      5, 1, 2, 5, 2, 6,
      0, 4, 7, 0, 7, 3,
      7, 6, 2, 7, 2, 3,
      0, 1, 5, 0, 5, 4,
    ]);

    this.meshData = {
      vertices: positions,
      indices,
      vertexCount: positions.length / 3,
    };
  }

  async initComputePipeline() {
    const device = gpuManager.device;
    const shaderModule = device.createShaderModule({ code: COMPUTE_SHADER });
    this.computePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'main' },
    });
  }

  createBuffers() {
    const device = gpuManager.device;
    const vertexData = this.meshData.vertices;

    if (this.inputBuffer) this.inputBuffer.destroy();
    if (this.outputBuffer) this.outputBuffer.destroy();
    if (this.uniformBuffer) this.uniformBuffer.destroy();

    this.inputBuffer = device.createBuffer({
      size: vertexData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.inputBuffer.getMappedRange()).set(vertexData);
    this.inputBuffer.unmap();

    this.outputBuffer = device.createBuffer({
      size: vertexData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  setupBindGroup() {
    this.bindGroup = gpuManager.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.inputBuffer } },
        { binding: 1, resource: { buffer: this.outputBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffer } },
      ],
    });
  }

  async processOperation(operation, param) {
    const device = gpuManager.device;
    const opIndex = OP_MAP[operation] ?? 0;
    const vertexCount = this.meshData.vertexCount;

    const uniformData = new ArrayBuffer(16);
    const view = new DataView(uniformData);
    view.setUint32(0, opIndex, true);
    view.setFloat32(4, param, true);
    view.setUint32(8, vertexCount, true);
    view.setUint32(12, 0, true);
    device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    const workgroups = Math.ceil(vertexCount / 64);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.computePipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();

    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const elapsed = performance.now() - start;

    return elapsed;
  }

  async getProcessedVertices() {
    const size = this.meshData.vertices.byteLength;
    return gpuManager.readBuffer(this.outputBuffer, size);
  }
}

export const stlOps = new StlOperations();
