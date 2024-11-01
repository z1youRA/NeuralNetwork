<template>
	<div class="flow-graph">
		<!-- Add your template content here -->
	</div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { useComputeGraphStore } from '../store/computeGraphStore';

async function fetchNodes(modelName) {
	if (modelName === '') {
		return [];
	}

	try {
		const response = await fetch('/Models/' + modelName + '/nodes.json');
		console.log(response);
		const nodes = await response.json();
		return nodes;
	} catch (error) {
		console.error(error);
	}
}

const modelName = ref('Classification1raw');
//console log fetchNodes(modelName.value);
onMounted(async () => {
	nodes.value = await fetchNodes(modelName.value);
});

const computeGraphStore = useComputeGraphStore();
const nodes = ref([]);
const edges = ref([]);
function Flow() {
	fetchNodes(modelName.value).then((nodes) => {});
}
</script>

<style scoped></style>
