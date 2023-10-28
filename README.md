# videosd-lcm

LCM diffusion pipeline, combined with webrtc frontend for camera input and speech recognition.

![videosd](https://user-images.githubusercontent.com/36767/219042235-6585f79c-13a5-4380-a8b5-5e0ac3fc5733.gif)

# Run (dev)

```
docker compose up -d
docker compose exec backend-dev bash
root# python3 server.py
```

Open browser at localhost:80


# Config
Set configuration in diffusert/config.yml

```
gpus: 4
```

