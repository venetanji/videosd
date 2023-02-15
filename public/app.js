(() => {
    // The width and height of the captured photo. We will set the
    // width to the value defined here, but the height will be
    // calculated based on the aspect ratio of the input stream.
  
    const width = 512; // We will scale the photo width to this
    let height = 0; // This will be computed based on the input stream
  
    // |streaming| indicates whether or not we're currently streaming
    // video from the camera. Obviously, we start at false.
  
    let streaming = false;
    let host = window.location.hostname;
  
    // The various HTML elements we need to configure or control. These
    // will be set by the startup() function.
  
    let video = null;
    let canvas = null;
    let photo = null;
    let startbutton = null;
    let generating = false;
    const promptelement = document.getElementById("prompt");
    const strength = document.getElementById("strength");
    // add listener to promptelement on input


    var pc = null;
    var dc = null;

    function negotiate() {
        return pc.createOffer().then(function(offer) {
            return pc.setLocalDescription(offer);
        }).then(function() {
            // wait for ICE gathering to complete
            return new Promise(function(resolve) {
                if (pc.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    function checkState() {
                        if (pc.iceGatheringState === 'complete') {
                            pc.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    }
                    pc.addEventListener('icegatheringstatechange', checkState);
                }
            });
        }).then(function() {
            var offer = pc.localDescription;
            offer.sdp = sdpFilterCodec('video', "H264/90000", offer.sdp)
            console.log(offer.sdp)
            options = {
              "prompt": promptelement.value,
              "strength": parseFloat(strength.value)
            }
           
            return fetch(`http://${host}:8080/offer`, {
                body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type,
                    options: options
                }),
                headers: {
                    'Content-Type': 'application/json'
                },
                method: 'POST'
            });
        }).then(function(response) {
            return response.json();
        }).then(function(answer) {
            return pc.setRemoteDescription(answer);
        }).catch(function(e) {
            alert(e);
        });
    }

    function start(stream) {
        var config = {
            sdpSemantics: 'unified-plan'
        };

        //config.iceServers = [{urls: ['stun:stun.l.google.com:19302']}];
        
        pc = new RTCPeerConnection(config);

        promptdc = pc.createDataChannel('prompt', {"ordered": true});
        promptelement.oninput = function() {
          promptdc.send(JSON.stringify({prompt: promptelement.value}))
        };
        strength.oninput = function() {
          promptdc.send(JSON.stringify({strength: strength.value}))
        }

        recorddc = pc.createDataChannel('record', {"ordered": true});
        recorddc.onmessage = function(evt) {
          promptelement.value = evt.data
          promptelement.oninput()
          console.log(evt.data)
        };

        // connect audio / video
        pc.addEventListener('track', function(evt) {
            if (evt.track.kind == 'video') {
                document.getElementById('genvideo').srcObject = evt.streams[0];
            } else {
                document.getElementById('audio').srcObject = evt.streams[0];
            }
        });

        stream.getTracks().forEach(function(track) {
          pc.addTrack(track, stream);
        });
        negotiate();

    }

    function stop() {
        document.getElementById('stop').style.display = 'none';

        setTimeout(function() {
            pc.close();
        }, 500);
    }

  

    function startup() {

      video = document.getElementById("video");
      canvas = document.getElementById("canvas");
      photo = document.getElementById("genvideo");
      const startbutton = document.getElementById("startbutton");
      const recordButton = document.getElementById('record');

      navigator.mediaDevices
        .getUserMedia({ video: {facingMode: 'environment'}, audio: true })
        .then((stream) => {
          video.srcObject = stream;
          video.play();
          start(stream)

          recordButton.addEventListener('mousedown', function() {
            recorddc.send("start")
          });

          recordButton.addEventListener('touchstart', function() {
            recorddc.send("start")
          });
      
          recordButton.addEventListener('mouseup', function() {
            recorddc.send("stop")
          });

          recordButton.addEventListener('touchend', function() {
            recorddc.send("stop")
          });
      
        })
        .catch((err) => {
          console.error(`An error occurred: ${err}`);
        });
  
  

    startbutton.addEventListener(
      "click",
      (ev) => {
        document.getElementById('gen').requestFullscreen()
          .then(function() {
            screen.orientation.lock('landscape');
          })
          .catch(function(error) {
            // element could not enter fullscreen mode
          });
        generating = true;
        //generate();
        ev.preventDefault();
        startbutton.style.display = "none";
      },
      false
    );

    addEventListener('fullscreenchange', (event) => { 
        if (!document.fullscreenElement) {
            startbutton.style.display = "block";
        }
    });

 }


  function sdpFilterCodec(kind, codec, realSdp) {
    var allowed = []
    var rtxRegex = new RegExp('a=fmtp:(\\d+) apt=(\\d+)\r$');
    var codecRegex = new RegExp('a=rtpmap:([0-9]+) ' + escapeRegExp(codec))
    var videoRegex = new RegExp('(m=' + kind + ' .*?)( ([0-9]+))*\\s*$')
    
    var lines = realSdp.split('\n');

    var isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var match = lines[i].match(codecRegex);
            if (match) {
                allowed.push(parseInt(match[1]));
            }

            match = lines[i].match(rtxRegex);
            if (match && allowed.includes(parseInt(match[2]))) {
                allowed.push(parseInt(match[1]));
            }
        }
    }

    var skipRegex = 'a=(fmtp|rtcp-fb|rtpmap):([0-9]+)';
    var sdp = '';

    isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var skipMatch = lines[i].match(skipRegex);
            if (skipMatch && !allowed.includes(parseInt(skipMatch[2]))) {
                continue;
            } else if (lines[i].match(videoRegex)) {
                sdp += lines[i].replace(videoRegex, '$1 ' + allowed.join(' ')) + '\n';
            } else {
                sdp += lines[i] + '\n';
            }
        } else {
            sdp += lines[i] + '\n';
        }
    }

    return sdp;
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

  // Set up our event listener to run the startup process
  // once loading is complete.
  window.addEventListener("load", startup, false);
})();