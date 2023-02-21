# videosd

TensorRT accelerated diffusion pipeline, combined with webrtc frontend for camera input and speech recognition.

![videosd](https://user-images.githubusercontent.com/36767/219042235-6585f79c-13a5-4380-a8b5-5e0ac3fc5733.gif)

# Compile the engines

```
docker compose build backend
docker compose run --rm compile
```

This will run compile.py inside the container to generate the engine files inside the dedicated volumes. This may take a while depending on your GPU and available RAM.
Note that docker will kill the process if it exceeds allocated ram (in WSL this is 50% of your total ram, 8GB will not be enough).
Minimal VRAM tested on is 6GB on my RTX3060 laptop.  You can try the `python compile.py --disable-onnx-optimization` if it fails with the unet.
If you want to recompile, just rename or recreate the volumes or spin up the dev container, get a shell and run it from there. Good Luck.

# Run the server in production mode

```
docker compose --profile production up -d
```

App should be running on port 80. You can run the frontend locally at localhost:80 or run `ngrok http 80` and open the link from a mobile phone in the same network.

# Run the server in dev mode

```
docker compose --profile dev up -d
docker exec -it videosd-backend-dev-1 bash
```

You can now attach vscode to the dev container by browsing remotes or using the docker vscode extension. To start the server run `python server.py` within the container.
