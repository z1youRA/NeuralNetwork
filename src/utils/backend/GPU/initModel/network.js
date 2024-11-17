export async function postGradients(url, client_id, gradient, iteration) {
	try {
		const response = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'cache-control': 'no-cache',
			},
			body: JSON.stringify({
				client_id: client_id,
				gradient: gradient,
				round_id: iteration,
			}),
		});
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}
		let responseJson = await response.json();
		return responseJson;
	} catch (error) {
		console.error('Error:', error);
	}
}

export async function checkRoundStatus(url, iteration) {
	try {
		const response = await fetch(`${url}/?round_id=${iteration}`, {
			method: 'GET',
			headers: {
				'cache-control': 'no-cache',
			},
		});
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}
		let responseJson = await response.json();
		return responseJson;
	} catch (error) {
		console.error('Error:', error);
	}
}

export async function getNewGradient(url) {
	try {
		const response = await fetch(url, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
				'cache-control': 'no-cache',
			},
		});
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}
		let responseJson = await response.json();
		return responseJson;
	} catch (error) {
		console.error('Error:', error);
	}
}
