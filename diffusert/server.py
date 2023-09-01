import argparse
import asyncio
import json
import logging
import io
import os
import ssl
import uuid
import yaml
import torch
import random
from scipy import io as sio
import numpy as np
from cuda import cuda, nvrtc, cudart
import tensorrt as trt
from utilities import TRT_LOGGER


import time
#import whisper
config = yaml.safe_load(open("config.yaml"))
gpu_num = config['gpus']
from PIL import Image
from videopipeline import VideoSDPipeline 
from aiohttp import web, ClientSession
import aiohttp_cors
from av import VideoFrame, AudioFifo

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
bh = MediaBlackhole()
videosd = None

class STTTrack(MediaStreamTrack):
    """
    A track that receives audio frames from an another track and
    sends them to a speech-to-text service.
    """

    kind = "audio"

    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track
        self.recording = False
        self.recorder = AudioFifo()
        self.text = None

    def transcribe(self):

        samples = self.recorder.read_many(1024)
        if len(samples) == 0:
            return
        sample_rate = samples[0].sample_rate
        samples = np.array([x.to_ndarray() for x in samples])
        samples = samples.flatten()
        
        sio.wavfile.write('/tmp/prompt.wav',sample_rate*2, samples)
        self.audiofile = io.open('/tmp/prompt.wav','rb')
        self.recorder = AudioFifo()

    
    async def fetch(self):
        print("fetch")
        async with ClientSession() as session:
            url = 'http://whisper:9000/asr?task=transcribe&language=en&output=json'
            async with session.post(url, data={'audio_file':self.audiofile}) as response:
                print(response)
                response = await response.json(content_type='text/plain')
                self.text = response['text']
                return response['text']

    async def recv(self):
        frame = await self.track.recv()
        if self.recording:
            reframe = frame
            reframe.pts = None
            self.recorder.write(reframe)
        return frame


class VideoSDTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"
    

    def __init__(self, track, options):
        super().__init__()  # don't forget this!
        self.track = track
        self.options = options
        self.generating = [False for i in range(gpu_num)]
        self.last_gen_start = [time.time() for i in range(gpu_num)]
        self.last_gen_frame = time.time()
        self.avg_gen_time = 0.4
        #initialize the frame as black empty
        self.img = Image.new('RGB', (512, 512), (0, 0, 0))
        self.current_frame = None
        self.gen_task = None
    
    def diffuse(self,frame,gpu=0):
        print(self.options)
        cudart.cudaSetDevice(gpu)
        imgs = trt_models[gpu].infer(frame.to_image(),[self.options['prompt']],
            #prompt=,
            num_of_infer_steps = self.options['steps'],
            guidance_scale = self.options['guidance_scale'],
            strength = self.options['strength'],
            current_frame = self.current_frame.to_image()
            )
        self.generating[gpu] = False
        self.avg_gen_time = 0.5*self.avg_gen_time + 0.5*(time.time() - self.last_gen_start[gpu])
        print("Average gen time:", self.avg_gen_time)
        #self.img.paste(imgs[0],(gpu*imgs[0].width,gpu*imgs[0].height))
        self.current_frame = VideoFrame.from_image(imgs[0])


    async def recv(self):
        frame = await self.track.recv()
        for gpu in range(gpu_num):
            if not self.generating[gpu]:
                if not self.current_frame:
                    self.current_frame = frame
                if time.time() - np.max(self.last_gen_start) < self.avg_gen_time/gpu_num: break
                self.generating[gpu] = True
                self.last_gen_start[gpu] = time.time()
                if not self.current_frame:
                    self.current_frame = frame
                print("Generating on GPU ", gpu)
                asyncio.get_running_loop().run_in_executor(None, self.diffuse,frame,gpu)
                break
        self.current_frame.pts = frame.pts
        self.current_frame.time_base = frame.time_base
        return self.current_frame

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)
    prompt = ["A photo of a cat"]
    tracks = {'audio': None, 'video': None}
    spoken_prompt = AudioFifo()


    @pc.on("datachannel")
    def on_datachannel(channel):
        print(channel.label)
        if channel.label == "prompt":
            @channel.on("message")
            def on_message(message):
                message = json.loads(message)
                if 'strength' in message:
                    message['strength'] = float(message['strength'])
                if 'steps' in message:
                    message['steps'] = int(message['steps'])
                if 'guidance_scale' in message:
                    message['guidance_scale'] = float(message['guidance_scale'])
                for key, value in message.items():
                    tracks['video'].options[key] = value

                print(message)
        elif channel.label == "record":
            @channel.on("message")
            def on_message(message):
                if message == "start":
                    tracks['audio'].recording = True
                    print("start recording")
                elif message == "stop":
                    tracks['audio'].recording = False
                    yield from asyncio.get_running_loop().run_in_executor(None, tracks['audio'].transcribe)
                    task = asyncio.create_task(tracks['audio'].fetch())
                    task.add_done_callback(lambda x: channel.send(task.result()))
                    print("stop recording")



    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)
        
        if track.kind == "video":
            tracks['video'] = VideoSDTrack(track, params["options"])
            pc.addTrack(tracks['video'])

        if track.kind == "audio":
            tracks['audio'] = STTTrack(track)
            bh.addTrack(tracks['audio'])
            
            print('audiotrack')

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await bh.stop()

    # handle offer
    print(offer)
    await pc.setRemoteDescription(offer)
    await bh.start()

    # send answer
    answer = await pc.createAnswer()
    print(answer)
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    cors.add(app.router.add_post("/offer", offer))
    app.on_shutdown.append(on_shutdown)
    trt_models = [None for i in range(gpu_num)]

    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    
    # load trt model into every gpus
    # TODO make this async and return when all models are loaded
    
    for i in range(gpu_num):
        cudart.cudaSetDevice(i)
        trt_models[i] = VideoSDPipeline(device=i, scheduler="EulerA")
        trt_models[i].loadEngines(engine_dir=config['model'])
        trt_models[i].loadResources(360,640,1)

    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )

