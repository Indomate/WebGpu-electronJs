class GPUManager {
  constructor() {
    this.device = null;
    this.queue = null;
    this.adapter = null;
  }

  async init() {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported on this browser');
    }

    try {
      this.adapter = await navigator.gpu.requestAdapter();
      if (!this.adapter) {
        throw new Error('Could not request WebGPU adapter');
      }

      this.device = await this.adapter.requestDevice();
      this.queue = this.device.queue;

      return {
        success: true,
        info: this.adapter.name || 'Unknown GPU'
      };
    } catch (error) {
      console.error('WebGPU initialization failed:', error);
      throw error;
    }
  }

  createBuffer(data, usage) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      mappedAtCreation: true,
      usage
    });

    new Float32Array(buffer.getMappedRange()).set(new Float32Array(data));
    buffer.unmap();
    return buffer;
  }

  createShaderModule(code) {
    return this.device.createShaderModule({ code });
  }

  createComputePipeline(shaderModule, bindGroupLayout) {
    return this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
      }),
      compute: { module: shaderModule, entryPoint: 'main' }
    });
  }

  createBindGroupLayout(entries) {
    return this.device.createBindGroupLayout({ entries });
  }

  createBindGroup(layout, resources) {
    return this.device.createBindGroup({
      layout,
      entries: resources
    });
  }

  async readBuffer(buffer, size) {
    const stagingBuffer = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
    this.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingBuffer.getMappedRange()).slice();
    stagingBuffer.unmap();

    return result;
  }

  dispatchCompute(pipeline, bindGroup, workgroupCountX, workgroupCountY = 1, workgroupCountZ = 1) {
    const commandEncoder = this.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
    pass.end();

    this.queue.submit([commandEncoder.finish()]);
  }
}

const gpuManager = new GPUManager();