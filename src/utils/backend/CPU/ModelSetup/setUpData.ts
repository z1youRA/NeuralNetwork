import Data from '../tools/DataClass';
import { useComputeGraphStore } from '../../../../store/computeGraphStore';
import setUpModel from './setUpModel';
import { getClientId, fetchDataset } from '../tools/client';

const getCSV_classify = async () => {
	const computeGraphStore = useComputeGraphStore();

	// const datasetName = 'easy_class.';
	// startTrain(datasetName);

	const client_id = await getClientId();

	const dataset = await fetchDataset(client_id);

	// stopFlag.value = false;

	const dataArray: number[][] = dataset
		.trim()
		.split('\n')
		.map((row) => row.split(',').map(Number));

	const data = new Data(dataArray, 0, dataArray.length - 1, 2, 2, computeGraphStore.batchSize);
	data.dataSetName = 'classify';
	console.log('data is ready', dataArray.length);

	setUpModel(data);
	// 	},
	// });
};

export { getCSV_classify };
