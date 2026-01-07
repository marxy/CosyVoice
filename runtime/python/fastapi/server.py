# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


def unknown_args_to_kwargs(unknown_args):
    kwargs = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]

        # 情况1: --key=value 形式
        if arg.startswith('--') and '=' in arg:
            key, value = arg[2:].split('=', 1)
            key = key.replace('-', '_')
            kwargs[key] = _auto_convert(value)
            i += 1

        # 情况2: --key value 形式（flag 或带值）
        elif arg.startswith('--'):
            key = arg[2:].replace('-', '_')
            # 查看下一个参数是否存在且不是新 key
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                value = unknown_args[i + 1]
                kwargs[key] = _auto_convert(value)
                i += 2  # 消耗两个参数
            else:
                # 无后续值 → 当作 flag（True）
                kwargs[key] = True
                i += 1

        else:
            raise ValueError(f"孤立参数（不在 --key 后）: {arg}")

    return kwargs


def _auto_convert(s):
    """尝试将字符串自动转为 int/float/bool，失败则保留 str"""
    s = s.strip()
    # 先试 bool（避免 '1' 被转成 True）
    if s.lower() in ('true', 'false'):
        return s.lower() == 'true'
    # 再试 int
    try:
        if '.' not in s and 'e' not in s.lower():
            return int(s)
    except ValueError:
        pass
    # 再试 float
    try:
        return float(s)
    except ValueError:
        pass
    # 否则保留字符串
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--log_level',
                        type=str,
                        default='debug',
                        help='local logging level')
    args, unknown_args = parser.parse_known_args()
    extra_kwargs = unknown_args_to_kwargs(unknown_args)
    cosyvoice = AutoModel(model_dir=args.model_dir, **extra_kwargs)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level=args.log_level)
