import argparse
import asyncio
import json
import logging
import io
import os
import ssl
import uuid
from scipy import io as sio
import numpy as np
#import whisper



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
                response = await response.json()
                self.text = response['text']
                print(response)
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
        self.generating = False
        self.current_frame = None
        self.gen_task = None
    
    def diffuse(self,frame):
        print(self.options)
        imgs = trt_model.infer(
            prompt=[self.options['prompt']],
            num_of_infer_steps = 20,
            guidance_scale = 7,
            init_image= frame.to_image(),
            strength = self.options['strength'],
            seed=43)
        self.generating = False
        self.current_frame = VideoFrame.from_image(imgs[0])

    async def recv(self):
        frame = await self.track.recv()
        if not self.generating:
            self.generating = True
            if not self.current_frame:
                self.current_frame = frame
            asyncio.get_running_loop().run_in_executor(None, self.diffuse,frame)
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

    trt_model = VideoSDPipeline()
    trt_model.loadEngines()
    trt_model.loadModules()

    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )

