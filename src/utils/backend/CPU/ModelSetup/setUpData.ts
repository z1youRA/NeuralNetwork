import Papa from 'papaparse';
// import { ParseResult } from 'papaparse';
import Data from '../tools/DataClass';
import { useComputeGraphStore } from '../../../../store/computeGraphStore';
import setUpModel from './setUpModel';
import { getClientId, fetchDataset, startTrain } from '../tools/client';

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
	// const inputString =
	// 	dataset == 'easy_class.'
	// 		? '/Datasets/dataClass1.csv'
	// 		: dataset == 'medium_class.'
	// 			? '/Datasets/dataClass2.csv'
	// 			: dataset == 'hard_class.'
	// 				? '/Datasets/dataClass3.csv'
	// 				: '';

	// Papa.parse(inputString, {
	// 	header: false,
	// 	download: true,
	// 	skipEmptyLines: true,
	// 	delimiter: ',',
	// 	complete: (results: { data: unknown[] }) => {
	// 		const dataArray = results.data.map((row: unknown) =>
	// 			Object.values(row as { [key: string]: string }).map((value: string, index: number) => (index == 0 ? Number(value) : Number(value)))
	// 		);

	//#TODO Data length

	const data = new Data(dataArray, 0, dataArray.length - 1, 2, 2, computeGraphStore.batchSize);
	data.dataSetName = 'classify';
	console.log('data is ready', dataArray.length);

	setUpModel(data);
	// 	},
	// });
};

export { getCSV_classify };
