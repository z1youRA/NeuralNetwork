// import structs
import structs from '../wgsl_operations/structs.wgsl';

// forward computation imports
import copyInput from '../wgsl_operations/forwardOperations/copyInput.wgsl';
import forward from '../wgsl_operations/forwardOperations/forward.wgsl';
import addTensors from '../wgsl_operations/forwardOperations/addTensors.wgsl';
import correlateTensors from '../wgsl_operations/forwardOperations/correlateTensors.wgsl';
import multiplyTensors from '../wgsl_operations/forwardOperations/multiplyTensors.wgsl';
import ReLUTensor from '../wgsl_operations/forwardOperations/ReLUTensor.wgsl';
import softmaxTensor from '../wgsl_operations/forwardOperations/softmaxTensor.wgsl';
import CETensors from '../wgsl_operations/forwardOperations/CETensors.wgsl';
import MSETensor from '../wgsl_operations/forwardOperations/MSETensor.wgsl';
import OneHotTensor from '../wgsl_operations/forwardOperations/OneHotTensor.wgsl';
const forwards =
	copyInput + addTensors + correlateTensors + multiplyTensors + ReLUTensor + softmaxTensor + CETensors + OneHotTensor + MSETensor + forward;

// partial derivative computation imports
import computePartialDerivatives from '../wgsl_operations/partialDerivativesOperations/computePartialDerivatives.wgsl';
import pd_add from '../wgsl_operations/partialDerivativesOperations/pd_add.wgsl';
import pd_multiply from '../wgsl_operations/partialDerivativesOperations/pd_multiply.wgsl';
import pd_correlate from '../wgsl_operations/partialDerivativesOperations/pd_correlate.wgsl';
import pd_softmaxCE from '../wgsl_operations/partialDerivativesOperations/pd_softmaxCE.wgsl';
import pd_MSE from '../wgsl_operations/partialDerivativesOperations/pd_MSE.wgsl';
const partialDerivatives = pd_add + pd_multiply + pd_correlate + pd_softmaxCE + pd_MSE + computePartialDerivatives;

// partial derivative computation imports
import computeGradients from '../wgsl_operations/addGradientsOperations/computeGradients.wgsl';
import gr_add from '../wgsl_operations/addGradientsOperations/gr_add.wgsl';
import gr_multiply from '../wgsl_operations/addGradientsOperations/gr_multiply.wgsl';
import gr_correlate from '../wgsl_operations/addGradientsOperations/gr_correlate.wgsl';
import gr_ReLU from '../wgsl_operations/addGradientsOperations/gr_ReLU.wgsl';
import gr_softmaxCE from '../wgsl_operations/addGradientsOperations/gr_softmaxCE.wgsl';
import gr_MSE from '../wgsl_operations/addGradientsOperations/gr_MSE.wgsl';
const addGradients = gr_add + gr_multiply + gr_correlate + gr_ReLU + gr_softmaxCE + gr_MSE + computeGradients;

import updateData from '../wgsl_operations/updateDataOperations/updateData.wgsl';

import main from '../wgsl_operations/main.wgsl';

import { getxValues, getPredValues, getTrueValues, getErrorValue, getGradientValues } from './testSet.js';
import { ref } from 'vue';
import { useComputeGraphStore } from '../../../../store/computeGraphStore.js';
import { getNewGradient, checkRoundStatus, postGradients } from './network.js';

const stopFlag = ref(false);

function getStopFlag() {
	return stopFlag.value;
}

function setFlagTrain() {
	stopFlag.value = false;
}

function setFlagStop() {
	stopFlag.value = true;
}

async function MatMul(Offsets, FlatData, BackwardTape, GradientTape, _iterations, data, model, forwardTape, gradientTape, backwardTape) {
	const store = useComputeGraphStore();

	const numIterations = _iterations;
	const server_domain = 'http://localhost:8000';

	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		console.log('no adapter');
		return;
	}
	const device = await adapter.requestDevice();

	const gpuBufferOffsets = device.createBuffer({
		mappedAtCreation: true,
		size: Offsets.byteLength,
		usage: GPUBufferUsage.STORAGE,
	});
	const arrayBufferOffsets = new Float32Array(gpuBufferOffsets.getMappedRange());
	arrayBufferOffsets.set(Offsets);
	gpuBufferOffsets.unmap();

	const gpuBufferFlatData = device.createBuffer({
		mappedAtCreation: true,
		size: FlatData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
	});
	const arrayBufferFlatData = new Float32Array(gpuBufferFlatData.getMappedRange());
	arrayBufferFlatData.set(FlatData);
	gpuBufferFlatData.unmap();

	const gpuBufferBackwardTape = device.createBuffer({
		mappedAtCreation: true,
		size: BackwardTape.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	const arrayBufferBackwardTape = new Float32Array(gpuBufferBackwardTape.getMappedRange());
	arrayBufferBackwardTape.set(BackwardTape);
	gpuBufferBackwardTape.unmap();

	const gpuBufferGradientTape = device.createBuffer({
		mappedAtCreation: true,
		size: GradientTape.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});
	const arrayBufferGradientTape = new Float32Array(gpuBufferGradientTape.getMappedRange());
	arrayBufferGradientTape.set(GradientTape);
	gpuBufferGradientTape.unmap();

	const resultMatrixBuffer = device.createBuffer({
		mappedAtCreation: true,
		size: FlatData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});
	const resultMatrixArray = new Float32Array(resultMatrixBuffer.getMappedRange());
	resultMatrixArray.set(FlatData);
	resultMatrixBuffer.unmap();

	const controlBuffer = device.createBuffer({
		size: 20,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});

	data.shuffle();
	console.log(data.getInputData());
	let inputData = new Float32Array(data.getInputDataBuffer());

	const inputDataBuffer = device.createBuffer({
		size: inputData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, // GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC // | GPUBufferUsage.MAP_WRITE
	});

	let trueValues = new Float32Array(data.getTrueValuesAny());
	const trueValuesBuffer = device.createBuffer({
		size: trueValues.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, // GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC // | GPUBufferUsage.MAP_WRITE
	});

	let EmptyAccuracies = new Float32Array(numIterations);

	const gpuBufferAvgAccuracy = device.createBuffer({
		mappedAtCreation: true,
		size: EmptyAccuracies.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
	});
	const arrayBufferAvgAccuracy = gpuBufferAvgAccuracy.getMappedRange();
	new Float32Array(arrayBufferAvgAccuracy).set(EmptyAccuracies);
	gpuBufferAvgAccuracy.unmap();

	//init layout
	const bindGroupLayout = device.createBindGroupLayout({
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'read-only-storage',
				},
			},
			{
				binding: 1,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},

			{
				binding: 2,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},
			{
				binding: 3,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},
			{
				binding: 4,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},
			{
				binding: 5,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},
			{
				binding: 6,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},
			{
				binding: 7,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},
		],
	});

	const bindGroup = device.createBindGroup({
		layout: bindGroupLayout,
		entries: [
			{
				binding: 0,
				resource: {
					buffer: gpuBufferOffsets,
				},
			},
			{
				binding: 1,
				resource: {
					buffer: gpuBufferFlatData,
				},
			},
			{
				binding: 2,
				resource: {
					buffer: gpuBufferBackwardTape,
				},
			},
			{
				binding: 3,
				resource: {
					buffer: controlBuffer,
				},
			},
			{
				binding: 4,
				resource: {
					buffer: inputDataBuffer,
				},
			},
			{
				binding: 5,
				resource: {
					buffer: gpuBufferGradientTape,
				},
			},
			{
				binding: 6,
				resource: {
					buffer: trueValuesBuffer,
				},
			},
			{
				binding: 7,
				resource: {
					buffer: gpuBufferAvgAccuracy,
				},
			},
		],
	});

	const shaderModule = device.createShaderModule({
		code: structs + forwards + partialDerivatives + addGradients + updateData + main,
	});

	const computePipeline = device.createComputePipeline({
		layout: device.createPipelineLayout({
			bindGroupLayouts: [bindGroupLayout],
		}),
		compute: {
			module: shaderModule,
			entryPoint: 'main',
		},
	});

	const gpuReadBuffer = device.createBuffer({
		size: FlatData.byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	});

	const gpuReadAvgAccuracyBuffer = device.createBuffer({
		size: EmptyAccuracies.byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	});

	const gpuSetBuffer = device.createBuffer({
		size: FlatData.byteLength,
		usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
	});

	// var startTime = performance.now();
	let xValues_all = [];
	let predValues_all = [];
	let trueValues_all = [];
	let errorsArray = [];

	var numExtraIterations = 1;
	if (model.batchSize < 50) {
		numExtraIterations = 10;
	} else if (model.batchSize < 100) {
		numExtraIterations = 5;
	} else if (model.batchSize < 200) {
		numExtraIterations = 2;
	}
	const framerate = 20;

	console.log('enter iteration');
	for (let iteration = 0; iteration < numIterations + 3 * framerate; iteration++) {
		if (stopFlag.value == true) {
			return;
		}
		data.shuffle();

		inputData = new Float32Array(data.getInputDataBuffer());
		trueValues = new Float32Array(data.getTrueValuesAny());
		device.queue.writeBuffer(inputDataBuffer, 0, inputData.buffer, 0, inputData.byteLength);

		let inputTensorId = data.tensorInputId;
		let commandEncoder = device.createCommandEncoder();
		let passEncoder = commandEncoder.beginComputePass();
		let control = new Float32Array([inputTensorId, -1, -1, 0, iteration]);
		device.queue.writeBuffer(controlBuffer, 0, control.buffer, 0, control.byteLength);

		passEncoder.setPipeline(computePipeline);
		passEncoder.setBindGroup(0, bindGroup);
		let workgroupCountX = Math.ceil(model.tensors[inputTensorId].rows / 16);
		let workgroupCountY = Math.ceil(model.tensors[inputTensorId].cols / 16);
		passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
		passEncoder.end();
		let gpuCommands = commandEncoder.finish();
		device.queue.submit([gpuCommands]);

		//compute type 0 - update trueValues
		device.queue.writeBuffer(trueValuesBuffer, 0, trueValues.buffer, 0, trueValues.byteLength);
		let trueValuesTensorId = data.tensorTrueId;
		commandEncoder = device.createCommandEncoder();
		passEncoder = commandEncoder.beginComputePass();
		control = new Float32Array([trueValuesTensorId, -1, -1, 0, iteration]);
		device.queue.writeBuffer(controlBuffer, 0, control.buffer, 0, control.byteLength);
		passEncoder.setPipeline(computePipeline);
		passEncoder.setBindGroup(0, bindGroup);
		workgroupCountX = Math.ceil(model.tensors[trueValuesTensorId].rows / 16);
		workgroupCountY = Math.ceil(model.tensors[trueValuesTensorId].cols / 16);
		passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
		passEncoder.end();
		gpuCommands = commandEncoder.finish();
		device.queue.submit([gpuCommands]);

		//compute type 1 - forward
		const numInferences = forwardTape.length;
		for (let i = 0; i < numInferences; i++) {
			const curTensorId = forwardTape[i];

			// 创建 GPU 命令编码器和计算通道
			const commandEncoder = device.createCommandEncoder();
			const passEncoder = commandEncoder.beginComputePass();

			// 设置控制缓冲区，指定当前张量 ID 和计算类型
			let control = new Float32Array([curTensorId, -1, -1, 1, iteration]);
			device.queue.writeBuffer(controlBuffer, 0, control.buffer, 0, control.byteLength);

			// 设置计算管线和绑定组
			passEncoder.setPipeline(computePipeline);
			passEncoder.setBindGroup(0, bindGroup);

			// 计算工作组大小，确保覆盖所有数据
			let workgroupCountX = Math.ceil(model.tensors[curTensorId].rows / 16);
			let workgroupCountY = Math.ceil(model.tensors[curTensorId].cols / 16);

			// 执行计算
			passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
			passEncoder.end();

			// 提交 GPU 命令
			const gpuCommands = commandEncoder.finish();
			device.queue.submit([gpuCommands]);

			// 如果是最后一个张量并且需要读取数据，每一个iteration执行一次
			if (i === numInferences - 1 && iteration % framerate < numExtraIterations) {
				// 等待 GPU 执行完成
				await device.queue.onSubmittedWorkDone();

				// 创建新的命令编码器用于复制数据
				const readCommandEncoder = device.createCommandEncoder();
				readCommandEncoder.copyBufferToBuffer(
					gpuBufferFlatData, // 源缓冲区
					0, // 源偏移量
					gpuReadBuffer, // 目标缓冲区
					0, // 目标偏移量
					FlatData.byteLength // 数据大小
				);
				readCommandEncoder.copyBufferToBuffer(gpuBufferAvgAccuracy, 0, gpuReadAvgAccuracyBuffer, 0, EmptyAccuracies.byteLength);

				// 提交复制命令
				const readCommands = readCommandEncoder.finish();
				device.queue.submit([readCommands]);

				// 映射缓冲区以读取数据
				await gpuReadBuffer.mapAsync(GPUMapMode.READ);
				await gpuReadAvgAccuracyBuffer.mapAsync(GPUMapMode.READ);
				const arrayBuffer = new Float32Array(gpuReadBuffer.getMappedRange());

				// 处理预测值、真实值和误差
				predValues_all.push(getPredValues(arrayBuffer, model, Offsets));
				trueValues_all.push(getTrueValues(arrayBuffer, model, Offsets));
				errorsArray.push(getErrorValue(arrayBuffer, model, Offsets)); // 保存每次迭代的损失值
				xValues_all.push(getxValues(arrayBuffer, data, Offsets));

				// 解除映射以释放内存
				gpuReadBuffer.unmap();
				gpuReadAvgAccuracyBuffer.unmap();
			}
		}

		// 如果在指定的帧率间隔内，跳过后续处理以继续下一次迭代
		// if (iteration % framerate < numExtraIterations) {
		// 	continue;
		// }

		// 在特定的迭代次数，计算并输出平均误差
		if (iteration % framerate === numExtraIterations) {
			let xVals = [].concat(...xValues_all);
			let predVals = [].concat(...predValues_all);
			let trueVals = [].concat(...trueValues_all);
			let avgError = errorsArray.reduce((sum, error) => sum + error, 0) / errorsArray.length;

			if (iteration >= framerate - 1) {
				console.log('avgError', avgError, 'iteration', iteration);
				// set store for display
				// store.avgError = avgError;
				// store.predVals = predVals;
				// store.trueVals = trueVals;
				// store.xVals = xVals;
				store.setAvgError(avgError);
				store.setPredVals(predVals);
				store.setTrueVals(trueVals);
				store.setXVals(xVals);
				store.setModelIterations(iteration);
			}

			// 清空数据数组，准备下一轮数据收集
			predValues_all = [];
			trueValues_all = [];
			xValues_all = [];
			errorsArray = [];
		}

		// compute type 2 - compute partial derivatives
		// gradientTape [par1, child1, par2, child1, ...] in pairs
		const numPds = gradientTape.length / 2;
		for (let i = 0; i < numPds; i++) {
			const parTensorId = gradientTape[2 * i];
			const curTensorId = gradientTape[2 * i + 1];

			const commandEncoder = device.createCommandEncoder();
			const passEncoder = commandEncoder.beginComputePass();
			let control = new Float32Array([curTensorId, parTensorId, -1, 2, iteration]);
			device.queue.writeBuffer(controlBuffer, 0, control.buffer, 0, control.byteLength);
			passEncoder.setPipeline(computePipeline);
			passEncoder.setBindGroup(0, bindGroup);

			let workgroupCountX = 1;
			let workgroupCountY = 1;

			if (model.tensors[curTensorId].type == 1) {
				workgroupCountX = Math.ceil(model.tensors[curTensorId].rows / 16);
				workgroupCountY = Math.ceil(model.tensors[curTensorId].cols / 16);
			} else if (model.tensors[curTensorId].type == 2) {
				let isRightMultiplicator = model.tensors[parTensorId].isRightMultiplicator;
				if (isRightMultiplicator) {
					workgroupCountX = Math.ceil(model.tensors[curTensorId].rows / 16);
					workgroupCountY = Math.ceil(model.tensors[parTensorId].rows / 16);
				} else {
					workgroupCountX = Math.ceil(model.tensors[parTensorId].cols / 16);
					workgroupCountY = Math.ceil(model.tensors[curTensorId].cols / 16);
				}
			} else if (model.tensors[curTensorId].type == 3) {
				// ReLU
				// workgroupCountX = Math.ceil(  model.tensors[ curTensorId ].rows / 16);
				// workgroupCountY = Math.ceil(  model.tensors[ parTensorId ].cols / 16);
			} else if (model.tensors[curTensorId].type == 4) {
				//softmax
				workgroupCountX = Math.ceil(model.tensors[parTensorId].rows / 16);
				workgroupCountY = Math.ceil(model.tensors[parTensorId].cols / 16);
			} else if (model.tensors[curTensorId].type == 5) {
				// CE
				workgroupCountX = Math.ceil(model.tensors[parTensorId].rows / 16);
				workgroupCountY = Math.ceil(model.tensors[parTensorId].cols / 16);
			} else if (model.tensors[curTensorId].type == 7) {
				// CE
				workgroupCountX = Math.ceil(model.tensors[parTensorId].rows / 16);
				workgroupCountY = Math.ceil(model.tensors[parTensorId].cols / 16);
			}

			passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
			passEncoder.end();

			let gpuCommands = commandEncoder.finish();
			device.queue.submit([gpuCommands]);
		}

		const numGrds = gradientTape.length / 2;

		for (let i = 0; i < numGrds; i++) {
			const currTensorId = gradientTape[2 * i];
			const currChildId = gradientTape[2 * i + 1];

			let commandEncoder = device.createCommandEncoder();
			let passEncoder = commandEncoder.beginComputePass();
			let control = new Float32Array([
				currTensorId,
				-1 /* curr parent of interest */,
				currChildId /*curr child of interest */,
				3 /* compute type */,
				iteration,
			]);
			device.queue.writeBuffer(controlBuffer, 0, control.buffer, 0, control.byteLength);
			passEncoder.setPipeline(computePipeline);
			passEncoder.setBindGroup(0, bindGroup);

			let workgroupCountX = Math.ceil(model.tensors[currTensorId].rows / 16);
			let workgroupCountY = Math.ceil(model.tensors[currTensorId].cols / 16);
			passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
			passEncoder.end();

			let gpuCommands = commandEncoder.finish();
			device.queue.submit([gpuCommands]);
		}

		// UPLOAD gradients
		commandEncoder = device.createCommandEncoder();
		commandEncoder.copyBufferToBuffer(gpuBufferFlatData, 0, gpuReadBuffer, 0, FlatData.byteLength);
		gpuCommands = commandEncoder.finish();
		device.queue.submit([gpuCommands]);

		await gpuReadBuffer.mapAsync(GPUMapMode.READ);
		const dataReadBuffer = new Float32Array(gpuReadBuffer.getMappedRange());
		const gradientValues = getGradientValues(dataReadBuffer, model, Offsets);

		//POST gradients
		const responseJson = await postGradients(`${server_domain}/submit_gradients`, localStorage.getItem('client_id'), gradientValues, iteration);

		while (responseJson.status == 'waiting') {
			//wait 1s
			await new Promise((resolve) => setTimeout(resolve, 100));

			if (stopFlag.value == true) return; // avoid infinite loop

			//Query round status
			const responseJson = await checkRoundStatus(`${server_domain}/check_round_status`, iteration);
			console.log('LOG: Waiting for other clients: ', responseJson);
		}

		if (responseJson.status !== 'complete') {
			throw new Error('Error: Round not completed');
		}

		gpuReadBuffer.unmap();

		// get new gradient from server "/api/new-gradient"
		const responseNewGradJson = await getNewGradient(`${server_domain}/get_new_gradient`);

		const newGradientValues = responseNewGradJson.new_gradient;
		const flattenedGradientValues = newGradientValues.flat();
		// console.log('LOG: new gradient received', newGradientValues);

		const newGradientValuesBuffer = new Float32Array(flattenedGradientValues);

		// copy flatdata to unmap gpuSetBuffer
		const N = 15; // 每个张量在 Offsets 数组中占用的元素数量
		let gradientOffset = 0; // newGradientValuesBuffer 的偏移量
		const sourceBufferSize = newGradientValuesBuffer.byteLength; // 源数据缓冲区大小（字节）
		const destinationBufferSize = gpuBufferFlatData.size; // 目标缓冲区大小（字节）

		for (let tensor of model.tensors) {
			const rows = tensor.rows;
			const cols = tensor.cols;
			const tensorSizeInElements = rows * cols; // 张量的大小

			// 计算偏移量
			const baseIndex = 3 + tensor.id * N;
			const offsetDataIndex = baseIndex + 6;
			const flatDataOffset = Offsets[offsetDataIndex] * 4; // 数据在 FlatData 中的字节偏移量

			// 将梯度值写入到 gpuBufferFlatData 中
			if (gradientOffset + tensorSizeInElements * 4 > sourceBufferSize) {
				console.log('sourceBufferSize', sourceBufferSize);
				throw new Error('源数据的偏移量和大小超过了源缓冲区的大小。');
			}

			// 验证目标缓冲区大小
			if (flatDataOffset + tensorSizeInElements * 4 > destinationBufferSize) {
				throw new Error('目标缓冲区的偏移量和大小超过了目标缓冲区的大小。');
			}
			device.queue.writeBuffer(
				gpuBufferFlatData, // 目标缓冲区
				flatDataOffset, // 目标缓冲区的偏移量
				newGradientValuesBuffer, // 源数据缓冲区
				gradientOffset, // 源数据的偏移量
				tensorSizeInElements // 要写入的数据大小
			);
			// 更新偏移量
			gradientOffset += tensorSizeInElements;
		}

		// 确保写入操作完成
		await device.queue.onSubmittedWorkDone();

		// for debug
		// commandEncoder = device.createCommandEncoder();
		// commandEncoder.copyBufferToBuffer(gpuBufferFlatData, 0, gpuReadBuffer, 0, FlatData.byteLength);
		// gpuCommands = commandEncoder.finish();
		// device.queue.submit([gpuCommands]);

		// await gpuReadBuffer.mapAsync(GPUMapMode.READ);
		// const debugReadBuffer = new Float32Array(gpuReadBuffer.getMappedRange());
		// const updatedGradientData = getGradientValues(debugReadBuffer, model, Offsets);
		// console.log('LOG: updated gradient', updatedGradientData);
		// gpuReadBuffer.unmap();

		// compute type 4 - update data
		const numUpdates = backwardTape.length;
		for (let i = 3; i < numUpdates; ++i) {
			const currTensorId = backwardTape[i];

			let commandEncoder = device.createCommandEncoder();
			let passEncoder = commandEncoder.beginComputePass();
			let control = new Float32Array([
				currTensorId,
				-1 /* curr parent of interest */,
				-1 /*curr child of interest */,
				4 /* compute type */,
				iteration,
			]);
			device.queue.writeBuffer(controlBuffer, 0, control.buffer, 0, control.byteLength);
			passEncoder.setPipeline(computePipeline);
			passEncoder.setBindGroup(0, bindGroup);

			let workgroupCountX = Math.ceil(model.tensors[currTensorId].rows / 16);
			let workgroupCountY = Math.ceil(model.tensors[currTensorId].cols / 16);
			passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
			passEncoder.end();

			let gpuCommands = commandEncoder.finish();
			device.queue.submit([gpuCommands]);
		}
	}

	console.log('iteration complete');
	return;
}

export { MatMul, setFlagStop, setFlagTrain };
