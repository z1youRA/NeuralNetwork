from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict
from fastapi.responses import FileResponse
import numpy as np
import pandas
from pydantic import BaseModel
import uuid
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:5173", "http://localhost:5174"],  # 允许的源
	# allow_origins=["*"],  # 允许的源
	allow_credentials=True,
	allow_methods=["*"],  # 允许的 HTTP 方法
	allow_headers=["*"],  # 允许的 HTTP 头
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
	# 已经被删掉了，说明已经收到所有的梯度  check一定是发生在post梯度之后的
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
