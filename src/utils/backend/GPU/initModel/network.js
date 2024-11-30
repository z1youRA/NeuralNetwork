let ws = null;
let gradientCallbacks = new Map(); // 存储等待梯度响应的回调函数

// 添加重试连接逻辑
export async function initWebSocket(client_id, maxRetries = 3) {
	let retries = 0;

	while (retries < maxRetries) {
		try {
			console.log('Attempting to connect WebSocket...');
			ws = new WebSocket(`ws://localhost:8000/ws/${client_id}`);

			await new Promise((resolve, reject) => {
				const timeout = setTimeout(() => {
					reject(new Error('WebSocket connection timeout'));
				}, 5000);

				ws.onopen = () => {
					console.log('WebSocket connected successfully');
					clearTimeout(timeout);
					resolve();
				};

				ws.onerror = (error) => {
					console.error('WebSocket connection error:', error);
					clearTimeout(timeout);
					reject(error);
				};

				ws.onclose = (event) => {
					console.log('WebSocket connection closed', event.code, event.reason);
				};

				ws.onmessage = (event) => {
					const data = JSON.parse(event.data);
					if (data.type === 'new_gradient') {
						for (let callback of gradientCallbacks.values()) {
							callback(data.gradient);
						}
						gradientCallbacks.clear();
					}
				};
			});

			// 如果成功连接，跳出循环
			break;
		} catch (error) {
			retries++;
			console.log(`Connection attempt ${retries} failed:`, error);

			if (retries === maxRetries) {
				throw new Error(`WebSocket connection failed after ${maxRetries} retries: ${error.message}`);
			}

			// 增加重试间隔时间
			await new Promise((resolve) => setTimeout(resolve, 2000 * retries));
		}
	}
}

export async function postGradients(client_id, gradient, iteration) {
	if (!ws || ws.readyState !== WebSocket.OPEN) {
		throw new Error('WebSocket connection not established');
	}

	return new Promise((resolve) => {
		gradientCallbacks.set(iteration, resolve);
		ws.send(
			JSON.stringify({
				type: 'gradient',
				client_id: client_id,
				gradient: gradient,
				round_id: iteration,
			})
		);
	});
}

// 移除不再需要的函数
// checkRoundStatus() 和 getNewGradient() 不再需要
