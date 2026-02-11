export class GPUManager {
  constructor() {
    this.device = null;
    this.queue = null;
    this.adapter = null;
  }

  async init() {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not supported in this browser. Please use Chrome 113+ or Edge 113+.');
    }

    this.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!this.adapter) {
      throw new Error('Could not request WebGPU adapter. Check GPU drivers.');
    }

    this.device = await this.adapter.requestDevice();
    this.queue = this.device.queue;

    this.device.lost.then((info) => {
      console.error('WebGPU device lost:', info.message);
    });

    const adapterInfo = await this.adapter.requestAdapterInfo();
    return {
      info: adapterInfo.device || adapterInfo.description || 'WebGPU Active',
    };
  }

  createBuffer(data, usage) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      mappedAtCreation: true,
      usage,
    });
    new Float32Array(buffer.getMappedRange()).set(new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4));
    buffer.unmap();
    return buffer;
  }

  async readBuffer(srcBuffer, size) {
    const stagingBuffer = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(srcBuffer, 0, stagingBuffer, 0, size);
    this.queue.submit([encoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingBuffer.getMappedRange()).slice();
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    return result;
  }
}

export const gpuManager = new GPUManager();
