export function getGradientValues(arrayBuffer, model, Offsets) {
	const gradientValues = [];
	for (let tensor of model.tensors) {
		const tensorGradientValue = [];
		const tensorIndex = Offsets[3 + tensor.id * 15];
		const rows = arrayBuffer[tensorIndex + 3];
		const cols = arrayBuffer[tensorIndex + 4];
		const gradientIndex = Offsets[3 + tensor.id * 15 + 6];
		for (let i = 0; i < rows; i++) {
			for (let j = 0; j < cols; j++) {
				tensorGradientValue.push(arrayBuffer[gradientIndex + i * cols + j]);
			}
		}
		gradientValues.push(tensorGradientValue);
		// #TODO isGradient or not?
	}
	return gradientValues;
}

export function getxValues(arrayBuffer, data, Offsets) {
	const xValues = [];
	const t = data.tensorInputId;
	const tensorIndex = Offsets[3 + t * 15];
	const rows = arrayBuffer[tensorIndex + 3];
	const cols = arrayBuffer[tensorIndex + 4];
	const dataIndexStart = Offsets[3 + t * 15 + 5];
	for (let j = 0; j < cols; j++) {
		for (let i = 0; i < rows; i++) {
			// traverse the data in column major order to get (x, y) pairs
			xValues.push(arrayBuffer[dataIndexStart + i * cols + j]);
		}
	}

	return xValues;
}

export function getPredValues(arrayBuffer, model, Offsets) {
	const predValues = [];
	const t = model.outputTensor;
	const tensorIndex = Offsets[3 + t * 15];
	const rows = arrayBuffer[tensorIndex + 3];
	const cols = arrayBuffer[tensorIndex + 4];
	const dataIndexStart = Offsets[3 + t * 15 + 5];

	for (let j = 0; j < cols; j++) {
		for (let i = 0; i < rows; i++) {
			predValues.push(arrayBuffer[dataIndexStart + i * cols + j]);
		}
	}

	return predValues;
}

export function getTrueValues(arrayBuffer, model, Offsets) {
	const trueValues = [];
	const t = model.trueTensor;
	const tensorIndex = Offsets[3 + t * 15];
	const rows = arrayBuffer[tensorIndex + 3];
	const cols = arrayBuffer[tensorIndex + 4];
	const dataIndexStart = Offsets[3 + t * 15 + 5];
	for (let j = 0; j < cols; j++) {
		for (let i = 0; i < rows; i++) {
			trueValues.push(arrayBuffer[dataIndexStart + i * cols + j]);
		}
	}

	return trueValues;
}

export function getErrorValue(arrayBuffer, model, Offsets) {
	const t = model.lastTensor;
	const dataIndexStart = Offsets[3 + t * 15 + 5];

	return arrayBuffer[dataIndexStart];
}
