<script setup>
import { MatMul } from '/home/thierry/repos/neural_network_vue/neural_network/src/utils/backend/GPU/initModel/GPUTraining.js';
import SetUpModel from './utils/backend/CPU/ModelSetup/setUpModel.ts';
import FlowGraph from './components/FlowGraph.vue';
import { getCSV_classify } from './utils/backend/CPU/ModelSetup/setUpData.ts';
import { onMounted } from 'vue';

// onMounted(() => {
// 	console.log('mounted');
// 	getCSV_classify();
// });

getCSV_classify();
// MatMul();
</script>

<template>
	<div>
		<a href="https://vitejs.dev" target="_blank">
			<img src="/vite.svg" class="logo" alt="Vite logo" />
		</a>
		<a href="https://vuejs.org/" target="_blank">
			<img src="./assets/vue.svg" class="logo vue" alt="Vue logo" />
		</a>
		<!-- <SetUpModel /> -->
		<!-- <FlowGraph /> -->
	</div>
</template>
<style scoped>
.logo {
	height: 6em;
	padding: 1.5em;
	will-change: filter;
	transition: filter 300ms;
}
.logo:hover {
	filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vue:hover {
	filter: drop-shadow(0 0 2em #42b883aa);
}
</style>
