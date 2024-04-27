# videosd

A webrtc video interface to StableDiffusion pipelines. Built for https://backdropbuild.com/builds/v2/videosd

Check the youtube video below for a demo of how it works.


[![Videosd Demo](http://img.youtube.com/vi/1qMRPK1l_xw/0.jpg)](http://www.youtube.com/watch?v=1qMRPK1l_xw "Videosd Demo - Backdropbuild v2")


Live demo (1 old GPU): https://blendotron.art/


# Run (dev)

```
docker compose up -d
docker compose exec backend-dev bash
```

You can also attach vscode to the dev container by browsing remotes or using the docker vscode extension.
To start the server run `python server.py` within the container.

```
root# python3 server.py
```

# Config
Set configuration in diffusert/config.yml

```
gpus: 4
```

# Run the server in production mode

Edit `configs/turnserver.conf`

```
docker compose --profile production up -d
```

App should be running on port 80. You will need an ssl connection, for convenience you can use the certbot container or use ngrok (`ngrok http 80`)
