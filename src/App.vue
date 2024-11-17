<script setup>
import { startTrain } from './utils/backend/CPU/ModelSetup/setUpData.ts';
import { setFlagTrain, setFlagStop } from '/home/thierry/repos/neural_network_vue/neural_network/src/utils/backend/GPU/initModel/GPUTraining.js';
import {
	setReadyForTrain,
	getClientId,
	resetServer,
} from '/home/thierry/repos/neural_network_vue/neural_network/src/utils/backend/CPU/tools/client.ts';
import { ref } from 'vue';
import LossPlot from './components/LossPlot.vue';

function clearLocalStorage() {
	localStorage.clear();
}

const client_id = ref('');
const resetPlotFlag = ref(false);

async function getID() {
	client_id.value = await getClientId();
}
</script>

<template>
	<div>
		<button @click="getID">GET id: {{ client_id }}</button>
		<button @click="setReadyForTrain(client_id)">Ready</button>

		<button
			@click="
				{
					setFlagTrain(); // set stopFlag in GPUTraining.js to false
					startTrain(); // start training
				}
			"
		>
			Start Traning
		</button>
		<button
			@click="
				setFlagStop(); // set stopFlag in GPUTraining.js to true
				resetServer(); // reset server status
				resetPlotFlag = true; // reset plot
			"
		>
			STOP & RESET
		</button>

		<button @click="clearLocalStorage()">Clear Local ID</button>
		<LossPlot :reset-flag="resetPlotFlag" @resetComplete="resetPlotFlag = false" />
	</div>
</template>
