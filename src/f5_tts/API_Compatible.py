import os
import sys
from api import F5TTS
now_dir = os.getcwd()

import argparse
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
import uvicorn
from pydantic import BaseModel
# print(sys.path)


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()

# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv


APP = FastAPI()
class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k:int = 5
    top_p:float = 1
    temperature:float = 1
    text_split_method:str = "cut5"
    batch_size:int = 1
    batch_threshold:float = 0.75
    split_bucket:bool = True
    speed_factor:float = 1.0
    fragment_interval:float = 0.3
    seed:int = -1
    media_type:str = "wav"
    streaming_mode:bool = False
    parallel_infer:bool = True
    repetition_penalty:float = 1.35


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    global f5
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "weight path is required"})        
        f5=F5TTS(ckpt_file=weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    return JSONResponse(status_code=200, content={"message": "success"})

@APP.get("/tts")
async def tts_get_endpoint(
                        text: str = None,
                        text_lang: str = None,
                        ref_audio_path: str = None,
                        aux_ref_audio_paths:list = None,
                        prompt_lang: str = None,
                        prompt_text: str = "",
                        top_k:int = 5,
                        top_p:float = 1,
                        temperature:float = 1,
                        text_split_method:str = "cut0",
                        batch_size:int = 1,
                        batch_threshold:float = 0.75,
                        split_bucket:bool = True,
                        speed_factor:float = 1.0,
                        fragment_interval:float = 0.3,
                        seed:int = -1,
                        media_type:str = "wav",
                        streaming_mode:bool = False,
                        parallel_infer:bool = True,
                        repetition_penalty:float = 1.35
                        ):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size":int(batch_size),
        "batch_threshold":float(batch_threshold),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "fragment_interval":fragment_interval,
        "seed":seed,
        "media_type":media_type,
        "streaming_mode":streaming_mode,
        "parallel_infer":parallel_infer,
        "repetition_penalty":float(repetition_penalty)
    }
    return await tts_handle(req)
                

@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)

async def tts_handle(req:dict):
    global f5
    wav,sr,_=f5.infer(ref_text=req.get("prompt_text"),ref_file=req.get("ref_audio_path"),gen_text=req.get("text"),speed=req.get("speed_factor"))
    io_buffer = BytesIO()
    sf.write(io_buffer, wav, sr, format='wav')
    return Response(io_buffer.getvalue(), media_type=f"audio/wav")




if __name__ == "__main__":
    f5=F5TTS(ckpt_file="download\huggingface\hub\models--SWivid--F5-TTS\snapshots\d6bd6c3c3ec65c0a3ef25a6d3d09658c5e2817fd\F5TTS_v1_Base\model_1250000.safetensors")
    try:
        if host == 'None':
            host = None
        uvicorn.run(app=f'{os.path.basename(__file__).split(".")[0]}:APP',host=host, port=port, workers=1)
    except Exception as e:
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
