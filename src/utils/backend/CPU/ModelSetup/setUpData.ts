import Papa from 'papaparse';
// import { ParseResult } from 'papaparse';
import Data from '../tools/DataClass';
import { useComputeGraphStore } from '../../../../store/computeGraphStore';
import setUpModel from './setUpModel';

const dataset = 'easy_class.';

const getCSV_classify = () => {
	const computeGraphStore = useComputeGraphStore();
	console.log('getCSV called');

	const inputString =
		dataset == 'easy_class.'
			? '/Datasets/dataClass1.csv'
			: dataset == 'medium_class.'
				? '/Datasets/dataClass2.csv'
				: dataset == 'hard_class.'
					? '/Datasets/dataClass3.csv'
					: '';
	Papa.parse(inputString, {
		header: false,
		download: true,
		skipEmptyLines: true,
		delimiter: ',',
		complete: (results: { data: unknown[] }) => {
			const dataArray = results.data.map((row: unknown) =>
				Object.values(row as { [key: string]: string }).map((value: string, index: number) => (index == 0 ? Number(value) : Number(value)))
			);

			const data = new Data(dataArray, 2, 10000, 2, 2, computeGraphStore.batchSize);
			data.dataSetName = 'classify';
			console.log('data is ready');
			console.log(data.getInputData());

			setUpModel(data);
		},
	});
};

export { getCSV_classify };
