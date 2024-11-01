function generateNormalRandom(mean: number, stdDev: number) {
	const u = Math.random(); // uniformly distributed random number between 0 and 1
	const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
	return z * stdDev + mean; // transform the standard normal variable into a normal variable with mean and standard deviation
}

export class Tensor {
	// public
	id = 0;
	typestring: string;
	type = 0;
	// numDimensions = 2;
	// dimensions = [1,1];
	rows = 1;
	cols = 1;
	metaDims = [1];
	data: number[] = [];
	requiresGradient: boolean;
	isRightMultiplicator = false;
	gradientData: number[] = [];
	velocity_momentum: number[] = [];
	velocity_RMSProp: number[] = [];
	children: number[] = [];
	parents: number[] = [];
	partialDerivativeLeft: number[] = [];
	partialDerivativeRight: number[] = [];

	partner_rows = 1;
	partner_cols = 1;

	isInput = false;
	isOutput = false;
	isTrue = false;
	isLast = false;

	constructor(_type: string, _rows: number, _cols: number, _requiresGradient: boolean, _tensorId: number, _metaDims: Array<number> = []) {
		this.typestring = _type;
		this.setType(_type);
		this.rows = _rows;
		this.cols = _cols;
		this.metaDims = _metaDims;
		this.requiresGradient = _requiresGradient; // == true ? 1 : 0;
		this.data = new Array<number>(_rows * _cols).fill(0);
		// this.gradientData = _requiresGradient ? new Array<number>(_rows * _cols).fill(0) : new Array<number>(0);
		this.gradientData = new Array<number>(_rows * _cols).fill(0);
		this.velocity_momentum = new Array<number>(_rows * _cols).fill(0);
		this.velocity_RMSProp = new Array<number>(_rows * _cols).fill(0);
		this.id = _tensorId;
	}

	setType(type: string) {
		if (type == 'none') {
			this.type = 0;
		} else if (type == 'add') {
			this.type = 1;
		} else if (type == 'mult') {
			this.type = 2;
		} else if (type == 'ReLU') {
			this.type = 3;
		} else if (type == 'softmax') {
			this.type = 4;
		} else if (type == 'CE') {
			this.type = 5;
		} else if (type == 'OneHot') {
			this.type = 6;
		} else if (type == 'MSE') {
			this.type = 7;
		} else if (type == 'conv2D') {
			this.type = 8;
		}
	}

	addParent(_parent: number) {
		this.parents.push(_parent);
	}

	setParents(_parents: Array<number>) {
		this.parents = _parents;
	}

	addChild(_child: number) {
		this.children.push(_child);
	}

	setChildren(_children: Array<number>) {
		this.children = _children;
	}

	setRandomData() {
		const numEntries = this.rows * this.cols;
		for (let i = 0; i < numEntries; ++i) {
			this.data[i] = generateNormalRandom(0, Math.sqrt(2.0 / this.cols));
		}
	}
	//  	  generateNormalRandom(0, 0.5); //
	// Math.random() * 1.0 - 0.5;

	setAllOnes() {
		const numEntries = this.rows * this.cols;
		for (let i = 0; i < numEntries; ++i) {
			this.data[i] = 1;
		}
	}
}
