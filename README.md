# videosd

TensorRT accelerated diffusion pipeline, combined with webrtc frontend for camera input and speech recognition.

# Compile the engine

```
docker compose build
docker compose run --rm compile
```

This will run compile.py inside the container to generate the engine files inside the dedicated volumes. This may take a while depending on your GPU and available RAM.
Note that docker will kill the process if it exceeds allocated ram (in WSL this is 50% of your total ram, 8GB will not be enough).
Minimal VRAM tested on is 6GB on my 3060 mobile.  You can try the `python compile.py --disable-onnx-optimization` if it fails miserably with the unet.
If you want to recompile, just rename or recreate the volumes or spin up the dev container, get a shell and run it from there. Good Luck.

# Run the server in production mode

```
docker compose --profile production up -d
```

App should be running on port 80, change in compose if you need

# Run the server in dev mode

```
docker compose --profile dev up -d
npm install
node index.js
```

Serve the app with express on port 3000. You can then attach vscode to the dev container. To start the server run `python server.py` within the container.
