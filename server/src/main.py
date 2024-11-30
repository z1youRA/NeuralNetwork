from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import List, Dict, Set
from fastapi.responses import FileResponse
import numpy as np
import pandas
from pydantic import BaseModel
import uuid
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # 允许所有来源
	allow_credentials=True,
	allow_methods=["*"],  # 允许的 HTTP 方法
	allow_headers=["*"],  # 允许的 HTTP 头
	expose_headers=["*"]
)

csv_filename = '/home/thierry/repos/neural_network_vue/neural_network/server/public/Datasets/dataClass1.csv'

gradient_storage: Dict[int, Dict[str, List[List[float]]]] = {} # { round_id : {client_id: gradient} }
ready_clients: List[str] = []

EXPECTED_CLIENTS = 1

class GradientData(BaseModel):
	client_id: str
	gradient: List[List[float]]
	round_id: int

new_gradient: List[List[float]] = []
	
@app.get("/get_client_id")
async def get_client_id():
	return {"client_id": str(uuid.uuid4())}

@app.get("/ready_to_train/{client_id}")
async def ready_to_train(client_id: str):
	global EXPECTED_CLIENTS

	ready_clients.append(client_id)
	EXPECTED_CLIENTS = len(ready_clients)

	return {
		"status": "success",
		"message": "client is ready to train",
		"client_id": client_id
	}

def split_csv(file_path, total_part_num):
	df = pandas.read_csv(file_path)
	df_len = len(df)
	part_len = df_len // total_part_num
	for i in range(total_part_num):
		if i == total_part_num - 1:
			df_part = df.iloc[i * part_len:]
		else:
			df_part = df.iloc[i * part_len: (i + 1) * part_len]
		print("part", i, "len:", len(df_part))
		df_part.to_csv(file_path[:-4] + f'_part{i}.csv', index=False)

@app.get("/get_dataset/{client_id}")
async def get_dataset(client_id: str):
	if client_id not in ready_clients:
		return {
				"status": "error",
				"message": "client is not ready to train",
				"client_id": client_id
		}
	
	if len(ready_clients) == 0:
		return {
				"status": "error",
				"message": "client is not ready to train",
				"client_id": client_id
		}
	
	split_csv(csv_filename, len(ready_clients))
	client_part_index = ready_clients.index(client_id)
	part_filename = csv_filename[:-4] + f'_part{client_part_index}.csv'
	return FileResponse(part_filename, media_type='text/csv', filename=f'part{client_id}.csv')


@app.post("/submit_gradients")
async def submit_gradients(data: GradientData):
	global new_gradient
	
	round_id = data.round_id
	client_id = data.client_id
	gradient = data.gradient
	if client_id not in ready_clients:
		return {
				"status": "error",
				"message": "client is not ready to train",
				"client_id": client_id
		}

	# init gradient_storage[round_id]
	if round_id not in gradient_storage:
		gradient_storage[round_id] = {}

	gradient_storage[round_id][client_id] = gradient

	# print("gradient received from client: " + data.client_id + " len: "+ str(len(data.gradient)))

	# all gradients received
	if len(gradient_storage[round_id]) == len(ready_clients):
		# get all gradients from current round
		client_gradients = list(gradient_storage[round_id].values())
		new_gradient.clear()

		for tensors in zip(*client_gradients):
			tensor_avg = [sum(values) / len(ready_clients) for values in zip(*tensors)]
			new_gradient.append(tensor_avg)
		del gradient_storage[round_id]
		# print("new_gradient sent", len(new_gradient))
		return {
			"status": "complete",
			"message": "all gradients received",
			"round_id": round_id,
			"average_gradient": new_gradient
		}

	return {
		"status": "waiting",
		"round_id": round_id,
		"message": f"{len(ready_clients) - len(gradient_storage[round_id])} more clients needed"
	}

@app.get("/check_round_status/")
async def check_round_status(round_id: int):
	# 已经被删掉了，说明已经��到所有的梯度  check一定是发生在post梯度之后的
	if (round_id not in gradient_storage):
		return {
			"status": "complete",
			"message": "all gradients received",
			"round_id": round_id,
		}
	if (len(gradient_storage[round_id]) < len(ready_clients)):
		return {
			"status": "waiting",
			"round_id": round_id,
			"message": "waiting other clients to submit gradients"
		}
	else:
		return {
			"status": "complete",
			"message": "all gradients received",
			"round_id": round_id,
		}
	
@app.get("/get_new_gradient")
async def get_new_gradient():
	if len(new_gradient) == 0:
		return {"new_gradient": None}
	
	# print("new gradient sent", len(new_gradient))
	return {"new_gradient": new_gradient}

@app.get("/reset")
async def reset_server():
	ready_clients.clear()
	new_gradient.clear()
	gradient_storage.clear()

# WebSocket连接管理
class ConnectionManager:
	def __init__(self):
		self.active_connections: Dict[str, WebSocket] = {}  # client_id -> websocket
		
	async def connect(self, client_id: str, websocket: WebSocket):
		await websocket.accept()
		self.active_connections[client_id] = websocket
		
	def disconnect(self, client_id: str):
		if client_id in self.active_connections:
			del self.active_connections[client_id]
			
	async def broadcast_gradient(self, new_gradient: List[List[float]]):
		# 广播新梯度给所有连接的客户端
		message = {"type": "new_gradient", "gradient": new_gradient}
		for websocket in self.active_connections.values():
			await websocket.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    print(f"Attempting to connect client: {client_id}")
    try:
        await manager.connect(client_id, websocket)
        print(f"Client connected successfully: {client_id}")
        
        while True:
            try:
                data = await websocket.receive_json()
                # print(f"Received data from client {client_id}: {data['type']}")
                
                if data["type"] == "gradient":
                    round_id = data["round_id"]
                    gradient = data["gradient"]
                    
                    if round_id not in gradient_storage:
                        gradient_storage[round_id] = {}
                        
                    gradient_storage[round_id][client_id] = gradient
                    # print(f"Gradient received from client: {client_id}, length: {len(gradient)}")
                    
                    if len(gradient_storage[round_id]) == len(ready_clients):
                        client_gradients = list(gradient_storage[round_id].values())
                        new_gradient.clear()
                        
                        for tensors in zip(*client_gradients):
                            tensor_avg = [sum(values) / len(ready_clients) for values in zip(*tensors)]
                            new_gradient.append(tensor_avg)
                            
                        del gradient_storage[round_id]
                        # print(f"Broadcasting new gradient to all clients")
                        await manager.broadcast_gradient(new_gradient)
                        
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue
                
    except WebSocketDisconnect:
        print(f"Client disconnected: {client_id}")
        manager.disconnect(client_id)
        if client_id in ready_clients:
            ready_clients.remove(client_id)
    except Exception as e:
        print(f"Error in websocket connection: {e}")
        manager.disconnect(client_id)
        if client_id in ready_clients:
            ready_clients.remove(client_id)
