import { processors } from 'eslint-plugin-vue';
import { Tensor } from './TensorClass';

export class Model {
	// public
	tensors: Array<Tensor> = [];
	numTensors = 0;
	lastTensor = 0;
	// inputTensor : number = 0;
	learningRate = 0.3;
	momentum = 0;
	outputTensor = 0;
	trueTensor = 0;
	batchSize = 1;

	tensor(_type: string, _rows: number, _cols: number, _requiresGradient: boolean, initialization: string, _metaDims: Array<number> = []): Tensor {
		const tensor_new = new Tensor(_type, _rows, _cols, _requiresGradient, this.numTensors, _metaDims);

		// if (initialization == 'random') {
		// 	tensor_new.setRandomData();
		// }
		// if (initialization == 'ones') {
		// 	tensor_new.setAllOnew();
		// } else if (initialization == 'zeros') {
		// 	tensor_new.setAllZeros();
		// } // not needed, as the array is initialized with zeros

		// # TODO : numTensors is used as tensorId, but it is not incremented perheps because normal tensor is not linked to any neurons
		return tensor_new;
	}

	tensorAdd(tensor_left: Tensor, tensor_right: Tensor): Tensor {
		if (tensor_left.rows != tensor_right.rows || tensor_left.cols != tensor_right.cols) {
			alert('Tensor dimensions do not match');
		}
		this.numTensors++;
		const requiresGradient = true; ////left.requiresGradient || right.requiresGradient;
		const tensor_new = new Tensor('add', tensor_left.rows, tensor_right.cols, requiresGradient, this.numTensors, [tensor_left.rows]);
		tensor_new.addParent(tensor_left.id);
		tensor_new.addParent(tensor_right.id);
		tensor_left.addChild(tensor_new.id);
		tensor_right.addChild(tensor_new.id);
		this.tensors.push(tensor_new);

		return tensor_new;
	}

	tensorMult(tensor_left: Tensor, tensor_right: Tensor): Tensor {
		if (tensor_left.rows != tensor_right.cols) {
			alert('Tensor dimensions do not match');
		}
		this.numTensors++;
		const requiresGradient = true; ////left.requiresGradient || right.requiresGradient;
		const tensor_new = new Tensor('mult', tensor_left.rows, tensor_right.cols, requiresGradient, this.numTensors, [tensor_left.rows]);
		tensor_new.addParent(tensor_left.id);
		tensor_new.addParent(tensor_right.id);
		tensor_left.addChild(tensor_new.id);
		tensor_right.addChild(tensor_new.id);
		this.tensors.push(tensor_new);

		return tensor_new;
	}

	ReLU(tensor_parent: Tensor): Tensor {
		this.numTensors = this.numTensors + 1;
		const resRequiresGradient = true; //single.requiresGradient;
		const res = new Tensor('ReLU', tensor_parent.rows, tensor_parent.cols, resRequiresGradient, this.numTensors, [tensor_parent.rows]);
		res.addParent(tensor_parent.id);
		this.tensors[tensor_parent.id].addChild(this.numTensors);
		this.tensors.push(res);
		return res;
	}

	softmax(tensor_parent: Tensor): Tensor {
		this.numTensors = this.numTensors + 1;
		const resRequiresGradient = true; //single.requiresGradient;
		const res = new Tensor('softmax', tensor_parent.rows, tensor_parent.cols, resRequiresGradient, this.numTensors, [tensor_parent.rows]);
		res.addParent(tensor_parent.id);
		this.tensors[tensor_parent.id].addChild(this.numTensors);
		this.tensors.push(res);
		return res;
	}
}
