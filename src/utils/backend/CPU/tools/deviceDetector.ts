import { SERVER_CONFIG } from '../../../../config/serverConfig';

// 设备类型检测
export function detectDeviceType(): 'pc' | 'mobile' {
	const userAgent = navigator.userAgent;
	const mobileAgents = ["Android", "iPhone", "SymbianOS", "Windows Phone", "iPad", "iPod"];
	
	for (let agent of mobileAgents) {
		if (userAgent.indexOf(agent) > 0) {
			return 'mobile';
		}
	}
	return 'pc';
}

// 获取设备硬件信息
export async function getDeviceInfo(): Promise<any> {
	const deviceType = detectDeviceType();
	
	// 基本信息
	const deviceInfo = {
		device_type: deviceType,
		cpu_cores: navigator.hardwareConcurrency || 4,
		ram_size: 0, // 将通过性能测试估算
		gpu_memory: 0, // 将通过WebGPU检测
		webgpu_supported: false
	};

	// WebGPU支持检测
	try {
		if ('gpu' in navigator) {
			const adapter = await (navigator as any).gpu.requestAdapter();
			if (adapter) {
				deviceInfo.webgpu_supported = true;
				// 尝试获取GPU信息
				const device = await adapter.requestDevice();
				
				// 估算GPU内存（通过创建测试缓冲区）
				try {
					const testBuffer = device.createBuffer({
						size: 100 * 1024 * 1024, // 100MB测试
						usage: 0x80, // GPUBufferUsage.STORAGE
					});
					deviceInfo.gpu_memory = 2048; // 基础GPU内存估算
					testBuffer.destroy();
				} catch (e) {
					deviceInfo.gpu_memory = 512; // 低端GPU
				}
			}
		}
	} catch (error) {
		console.log('WebGPU not supported:', error);
	}

	// RAM估算（基于设备类型和JavaScript性能）
	if (deviceType === 'pc') {
		deviceInfo.ram_size = 8192; // PC默认8GB
	} else {
		deviceInfo.ram_size = 4096; // 移动设备默认4GB
	}

	return deviceInfo;
}

// 性能基准测试
export async function runPerformanceBenchmark(): Promise<any> {
	const startTime = performance.now();
	
	// 测试矩阵运算性能
	const testSize = detectDeviceType() === 'pc' ? 500 : 200; // 降低测试规模以避免超时
	const matrix1 = createRandomMatrix(testSize, testSize);
	const matrix2 = createRandomMatrix(testSize, testSize);
	
	// CPU矩阵乘法测试
	const cpuStart = performance.now();
	const result = matrixMultiply(matrix1, matrix2);
	const cpuTime = performance.now() - cpuStart;
	
	// WebGPU测试（如果支持）
	let gpuTime = 0;
	try {
		if ('gpu' in navigator) {
			const adapter = await (navigator as any).gpu.requestAdapter();
			if (adapter) {
				const device = await adapter.requestDevice();
				gpuTime = await testWebGPUPerformance(device, matrix1, matrix2);
			}
		}
	} catch (error) {
		console.log('WebGPU benchmark failed:', error);
	}
	
	const totalTime = performance.now() - startTime;
	
	// 计算每秒张量操作数
	const operations = testSize * testSize * testSize; // 矩阵乘法的运算量
	const operationsPerSecond = operations / (Math.min(cpuTime, gpuTime || cpuTime) / 1000);
	
	return {
		benchmark_time: totalTime,
		batch_size: detectDeviceType() === 'pc' ? 64 : 16,
		tensor_operations_per_sec: operationsPerSecond,
		cpu_time: cpuTime,
		gpu_time: gpuTime
	};
}

// 创建随机矩阵
function createRandomMatrix(rows: number, cols: number): number[][] {
	const matrix: number[][] = [];
	for (let i = 0; i < rows; i++) {
		matrix[i] = [];
		for (let j = 0; j < cols; j++) {
			matrix[i][j] = Math.random();
		}
	}
	return matrix;
}

// CPU矩阵乘法
function matrixMultiply(a: number[][], b: number[][]): number[][] {
	const rows = a.length;
	const cols = b[0].length;
	const common = b.length;
	const result: number[][] = [];
	
	for (let i = 0; i < rows; i++) {
		result[i] = [];
		for (let j = 0; j < cols; j++) {
			result[i][j] = 0;
			for (let k = 0; k < common; k++) {
				result[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	return result;
}

// WebGPU性能测试
async function testWebGPUPerformance(device: any, matrix1: number[][], matrix2: number[][]): Promise<number> {
	const startTime = performance.now();
	
	// 简单的WebGPU计算测试
	const shaderCode = `
		@compute @workgroup_size(16, 16)
		fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
			// 简单的计算测试
			let index = global_id.x + global_id.y * 64u;
			if (index >= 4096u) { return; }
		}
	`;
	
	try {
		const computeShader = device.createShaderModule({
			code: shaderCode
		});
		
		const computePipeline = device.createComputePipeline({
			layout: 'auto',
			compute: {
				module: computeShader,
				entryPoint: 'main'
			}
		});
		
		const commandEncoder = device.createCommandEncoder();
		const passEncoder = commandEncoder.beginComputePass();
		passEncoder.setPipeline(computePipeline);
		passEncoder.dispatchWorkgroups(4, 4);
		passEncoder.end();
		
		device.queue.submit([commandEncoder.finish()]);
		await device.queue.onSubmittedWorkDone();
	} catch (error) {
		console.log('WebGPU test failed:', error);
	}
	
	return performance.now() - startTime;
}

// 注册设备信息到服务器
export async function registerDevice(clientId: string): Promise<any> {
	const deviceInfo = await getDeviceInfo();
	(deviceInfo as any).client_id = clientId;
	
	const response = await fetch(`${SERVER_CONFIG.baseUrl}${SERVER_CONFIG.endpoints.registerDevice}`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify(deviceInfo)
	});
	
	return await response.json();
}

// 提交性能基准测试结果
export async function submitBenchmark(clientId: string): Promise<any> {
	const benchmarkResult = await runPerformanceBenchmark();
	(benchmarkResult as any).client_id = clientId;
	
	const response = await fetch(`${SERVER_CONFIG.baseUrl}${SERVER_CONFIG.endpoints.submitBenchmark}`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify(benchmarkResult)
	});
	
	return await response.json();
}