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
import sys
from scipy import io as sio
import numpy as np

import time
#import whisper 
from PIL import Image
from videopipeline import VideoSDPipeline 
from aiohttp import web, ClientSession
import aiohttp_cors
from av import VideoFrame, AudioFifo

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

logger = logging.getLogger("pc")
pcs = set()
sessions = 0
relay = MediaRelay()
bh = MediaBlackhole()
global session_watchdog
session_watchdog  = dict({'is_running': False})

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
        self.last_gen_start = [time.time() for i in range(gpu_num)]
        self.last_gen_frame = time.time()
        self.avg_gen_time = 0.4
        #initialize the frame as black empty

        self.init_frame = VideoFrame.from_ndarray(np.zeros((options['height'],options['width'],3),dtype=np.uint8))
        self.current_frame = None
        self.ref_frame = None
        self.gen_task = None
    
    async def diffuse(self,frame,gpu=0):

        self.last_gen_start[gpu] = time.time()
        try:
            img = await pipelines[gpu].infer.remote(frame.to_image(),**self.options)
        
        finally:
            generating[gpu] = False

        self.avg_gen_time = 0.95*self.avg_gen_time + 0.05*(time.time() - self.last_gen_start[gpu])
        sys.stdout.write("\rAverage gentime %f" % self.avg_gen_time)
        if self.options['ref']:
            self.ref_frame = img
        self.current_frame = VideoFrame.from_image(img)

    async def recv(self):
        frame = await self.track.recv()
        
        if not self.current_frame:
            self.current_frame = self.init_frame
            self.ref_frame = frame.to_image()
            if not session_watchdog['is_running']:
                session_watchdog['is_running'] = True
                print("Starting watchdog")
                asyncio.create_task(watchdog())
        


        for gpu in range(gpu_num):
            if not generating[gpu]:
                if time.time() - np.max(self.last_gen_start) < self.avg_gen_time*sessions/gpu_num: break
                generating[gpu] = True
                asyncio.create_task(self.diffuse(frame,gpu))
                break

        #with lock:
        outframe = self.current_frame
        outframe.pts = frame.pts
        outframe.time_base = frame.time_base
        return outframe

async def offer(request):

    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    turn = RTCIceServer(urls=["turn:blendotron.art:51820"], credential="blendotron", username="blendotron")

    config = RTCConfiguration(iceServers=[turn])

    pc = RTCPeerConnection(configuration=config)
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)
    tracks = {'audio': None, 'video': None}

    @pc.on("datachannel")
    def on_datachannel(channel):
        print(channel.label)
        if channel.label == "prompt":
            @channel.on("message")
            def on_message(message):
                message = json.loads(message)
                print("on message pre", message)
                if 'strength' in message:
                    message['strength'] = float(message['strength'])
                if 'steps' in message:
                    message['steps'] = int(message['steps'])
                if 'guidance_scale' in message:
                    message['guidance_scale'] = float(message['guidance_scale'])
                if 'controlnet_scale' in message:
                    message['controlnet_scale'] = float(message['controlnet_scale'])
                if 'style_fidelity' in message:
                    message['style_fidelity'] = float(message['style_fidelity'])
                if 'seed' in message:
                    message['seed'] = int(message['seed'])
                if 'ref' in message:
                    print(message['ref'])
                    message['ref'] = bool(message['ref'])
                if 'controlnet' in message:
                    message['controlnet'] = bool(message['controlnet']) 
                if 'set_ref' in message:
                    tracks['video'].ref_frame = tracks['video'].current_frame.to_image()
                    
                    #print("set ref bool in on message", message['set_ref'])


                for key, value in message.items():
                    tracks['video'].options[key] = value

                print("on message post", tracks['video'].options)

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
            pcs.discard(pc)
            await pc.close()
            await bh.stop()  
        if pc.connectionState == "closed":
            pcs.discard(pc)
            await pc.close()
            await bh.stop()
            
    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)
        if track.kind == "video":
            tracks['video'] = VideoSDTrack(track, params["options"])
            pc.addTrack(tracks['video'])

        if track.kind == "audio":
            tracks['audio'] = STTTrack(track)
            bh.addTrack(tracks['audio'])
            
            
        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            pcs.discard(pc)
            await bh.stop()
            await pc.close()

    @pc.on("close")
    def on_close():
        log_info("Close")
  
    # handle offer
    #print(offer)
    await pc.setRemoteDescription(offer)
    await bh.start()

    # send answer
    answer = await pc.createAnswer()
    #print(answer)
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
    config = yaml.safe_load(open("config.yaml"))
    gpu_num = config['gpus']

    global generating 
    generating = [False for i in range(gpu_num)]
    

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
    global pipelines 
    pipelines = [None for i in range(gpu_num)]
    
    for i in range(gpu_num):
        pipelines[i] = VideoSDPipeline.remote(**config)

    async def watchdog():
        while True:
            # print("Senders", senders)
            # print("Receivers", receivers)

            # count transports that are in connected state
            sessions = 0
            receivers_count = 0
            senders_count = 0
            for pc in pcs:
                for receiver in pc.getReceivers():
                    receivers_count += 1
                    sessions += 1 if receiver.transport.state == "connected" else 0
                for sender in pc.getSenders():
                    senders_count += 1

            if sessions == 0:
                for gpu in range(gpu_num):
                    generating[gpu] = False


            print("Receivers", receivers_count)
            print("Senders", senders_count)
            print("Active sessions", sessions)        
            print("Pcs", len(pcs))
            print("Generating", generating)
            await asyncio.sleep(5)

    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )

