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

const stopFlag = ref(false);

function getStopFlag() {
	return stopFlag.value;
}

function startTraining() {
	stopFlag.value = false;
}

function stopTraining() {
	stopFlag.value = true;
}

async function MatMul(
	Offsets,
	FlatData,
	BackwardTape,
	GradientTape,
	_iterations,
	data,
	model,
	forwardTape,
	gradientTape,
	backwardTape
	// stopFlag
) {
	// Offsets, FlatData, BackwardTape, GradientTape, _iterations, data, model
	// setAvgError,
	// 	setEdgesActive,
	// 	setXVals,
	// 	setTrueVals,
	// 	setPredVals,

	// 	forwardTape,
	// 	gradientTape,
	// 	backwardTape,
	// 	stopLearning,
	// 	setBreakTraining;

	//init data and other

	// console.log('webgpu starts');
	// console.log(model);

	const numIterations = _iterations;

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
			const commandEncoder = device.createCommandEncoder();
			const passEncoder = commandEncoder.beginComputePass();
			let control = new Float32Array([curTensorId, -1, -1, 1, iteration]);
			device.queue.writeBuffer(controlBuffer, 0, control.buffer, 0, control.byteLength);
			passEncoder.setPipeline(computePipeline);
			passEncoder.setBindGroup(0, bindGroup);
			let workgroupCountX = Math.ceil(model.tensors[curTensorId].rows / 16);
			let workgroupCountY = Math.ceil(model.tensors[curTensorId].cols / 16);
			passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
			passEncoder.end();

			if (iteration % framerate < numExtraIterations && i == numInferences - 1) {
				// Encode commands for copying buffer to buffer.
				commandEncoder.copyBufferToBuffer(
					gpuBufferFlatData /* source buffer */,
					0 /* source offset */,
					gpuReadBuffer /* destination buffer */,
					0 /* destination offset */,
					FlatData.byteLength /* size */
				);
				commandEncoder.copyBufferToBuffer(
					gpuBufferAvgAccuracy /* source buffer */,
					0 /* source offset */,
					gpuReadAvgAccuracyBuffer /* destination buffer */,
					0 /* destination offset */,
					EmptyAccuracies.byteLength /* size */
				);
				let gpuCommands = commandEncoder.finish();
				device.queue.submit([gpuCommands]);

				// Read buffer & setXVals, setTrueVals, setPredVals,
				await gpuReadBuffer.mapAsync(GPUMapMode.READ);
				await gpuReadAvgAccuracyBuffer.mapAsync(GPUMapMode.READ);
				const arrayBuffer = new Float32Array(gpuReadBuffer.getMappedRange());

				// getGradientValues(arrayBuffer, model, Offsets);

				predValues_all.push(getPredValues(arrayBuffer, model, Offsets));
				trueValues_all.push(getTrueValues(arrayBuffer, model, Offsets));
				errorsArray.push(getErrorValue(arrayBuffer, model, Offsets));
				xValues_all.push(getxValues(arrayBuffer, data, Offsets));
				gpuReadBuffer.unmap();
				gpuReadAvgAccuracyBuffer.unmap();
			} else {
				let gpuCommands = commandEncoder.finish();
				device.queue.submit([gpuCommands]);
			}
		}

		if (iteration % framerate < numExtraIterations) {
			continue;
		}
		if (iteration % framerate == numExtraIterations) {
			let xVals = [].concat(...xValues_all);
			let predVals = [].concat(...predValues_all);
			let trueVals = [].concat(...trueValues_all);
			let avgError = 0;

			for (let error of errorsArray) {
				avgError += error;
			}
			if (errorsArray.length > 0) {
				avgError /= errorsArray.length;
			}

			if (iteration >= framerate - 1) {
				// await setPredVals(predVals);
				// await setTrueVals(trueVals);
				// await setAvgError(avgError);
				// await setXVals(xVals);
				console.log('avgError', avgError, 'iteration', iteration);
			}

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

		console.log('LOG: old gradient', gradientValues);
		const postURL = `http://localhost:8000/submit_gradients`;

		const response = await fetch(postURL, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'cache-control': 'no-cache',
			},
			body: JSON.stringify({
				client_id: localStorage.getItem('client_id'),
				gradient: gradientValues,
				round_id: iteration,
			}),
		});
		let responseJson = await response.json();
		console.log(`LOG: Submit from client: ${localStorage.getItem('client_id')}`, responseJson);

		while (responseJson.status == 'waiting') {
			//wait 1s
			await new Promise((resolve) => setTimeout(resolve, 100));
			const response = await fetch(`http://localhost:8000/check_round_status/${iteration}`, {
				method: 'GET',
				headers: {
					'cache-control': 'no-cache',
				},
			});
			responseJson = await response.json();
			console.log('LOG: Waiting for other clients: ', responseJson);
		}

		if (responseJson.status !== 'complete') {
			throw new Error('Error: Round not completed');
		}

		gpuReadBuffer.unmap();

		// get new gradient from server "/api/new-gradient"
		const getURL = `http://localhost:8000/get_new_gradient`;
		const responseNewGrad = await fetch(getURL, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
				'cache-control': 'no-cache',
			},
		});
		const responseNewGradJson = await responseNewGrad.json();
		const newGradientValues = responseNewGradJson.new_gradient;
		const flattenedGradientValues = newGradientValues.flat();
		console.log('LOG: new gradient received', newGradientValues);

		const newGradientValuesBuffer = new Float32Array(flattenedGradientValues);

		console.log('buffer', newGradientValuesBuffer);
		console.log('buffer.buffer', newGradientValuesBuffer.buffer);

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
		commandEncoder = device.createCommandEncoder();
		commandEncoder.copyBufferToBuffer(gpuBufferFlatData, 0, gpuReadBuffer, 0, FlatData.byteLength);
		gpuCommands = commandEncoder.finish();
		device.queue.submit([gpuCommands]);

		await gpuReadBuffer.mapAsync(GPUMapMode.READ);
		const debugReadBuffer = new Float32Array(gpuReadBuffer.getMappedRange());
		const updatedGradientData = getGradientValues(debugReadBuffer, model, Offsets);
		console.log('LOG: updated gradient', updatedGradientData);
		gpuReadBuffer.unmap();

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

export { MatMul, stopTraining, startTraining, getStopFlag };
