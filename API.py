from utils.scripts import build_plot,build_ordered_sim_matrix,compose_new_simil, reduce_dimension
from utils.soap_descr import build_SOAP_descriptor, create_universe, pdb_parser
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import asyncio 
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import Response

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    pdb: str

class HeatValue(BaseModel):
    value: List[float]

class HeatMap(BaseModel):
    z: List[HeatValue]
    type: str
    x: List[str]
    y: List[str]

async def calculation_loop(pdblines):
    lines = await pdb_parser(pdblines)
    data= await build_SOAP_descriptor(create_universe(lines))
    data=data.todense()
    data=reduce_dimension(data)
    simil, ticks = build_plot(*build_ordered_sim_matrix(compose_new_simil(data)))
    return simil, ticks

@app.post("/api/compute_matrix" )
async def display(request: Item):
    try:
        simil, ticks = await calculation_loop(request.pdb)
    except Exception:
        return Response("Internal server error", status_code=500)
    simil = simil.tolist()
    simil=[list(map(lambda x: format(x, ".3f"), i)) for i in simil]
    ticks=ticks.tolist()
    return {"z": simil, "type": "heatmap", "x": ticks, "y": ticks}


@app.get("/api/retrieve_matrix")
async def read_root():
    simil, ticks = build_plot(*build_ordered_sim_matrix(compose_new_simil()))
    simil=simil.tolist()
    simil=[list(map(lambda x: format(x, ".3f"), i)) for i in simil]
    ticks=ticks.tolist()
    return {"z": simil, "type": "heatmap", "x": ticks, "y": ticks}



