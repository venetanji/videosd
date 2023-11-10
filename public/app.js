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
    const guidance_scale = document.getElementById("guidance_scale");
    const steps = document.getElementById("steps");
    const dropdown = document.getElementById("style_dropdown");
    const randomize_button = document.getElementById("randomize");
    const seed = document.getElementById("seed");
    const reference = document.getElementById("reference");
    const set_reference = document.getElementById("set_reference");
    const style_fidelity = document.getElementById("style_fidelity");
    const controlnet = document.getElementById("controlnet");


    randomize_button.onclick = function() {
        seed.value = Math.floor(Math.random() * 1000000000);
        seed.oninput()
    }

    // add listener to promptelement on input


    var pc = null;
    var dc = null;

    function updateTextarea() {
        
        promptelement.value = dropdown.options[dropdown.selectedIndex].text;
        
        // Dispatch an input event
        var event = new Event('input', {
            bubbles: true,
            cancelable: true,
        });
        promptelement.dispatchEvent(event);
    }

    dropdown.onchange = updateTextarea;


    function negotiate() {
        return pc.createOffer().then(function(offer) {
            return pc.setLocalDescription(offer);
        }).then(function() {
            // wait for ICE gathering to complete
            return new Promise(function(resolve) {
                if (pc.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    pc.addEventListener('icecandidate', function(event) {
                        console.log(event.candidate)
                        if (event.candidate && event.candidate.type === 'relay') {
                            resolve();
                        }
                    })
                }
            });
        }).then(function() {
            var offer = pc.localDescription;
            offer.sdp = sdpFilterCodec('video', "H264/90000", offer.sdp)
            console.log(offer.sdp)
            options = {
              "prompt": promptelement.value,
              "strength": parseFloat(strength.value),
              "guidance_scale": parseFloat(guidance_scale.value),
              "steps": parseInt(steps.value),
              "seed": parseInt(seed.value),
              "style_fidelity": 0.5,
              "ref": reference.checked,
              "controlnet": controlnet.checked
            }
           
            return fetch('/offer', {
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

    
        config.iceTransportPolicy = 'relay'

        config.iceServers = [{urls: ['turn:blendotron.art:51820?transport=udp'], username:'videosd',credential:'videosd'}];
        
        pc = new RTCPeerConnection(config);

        promptdc = pc.createDataChannel('prompt', {"ordered": true});
        promptelement.oninput = function() {
          promptdc.send(JSON.stringify({prompt: promptelement.value}))
        };
        strength.oninput = function() {
            document.getElementById("strengthval").innerHTML = strength.value;
            promptdc.send(JSON.stringify({strength: strength.value}))
        }
        guidance_scale.oninput = function() {
            document.getElementById("guidance_scaleval").innerHTML = guidance_scale.value;
            promptdc.send(JSON.stringify({guidance_scale: guidance_scale.value}))
          }
        steps.oninput = function() {
            document.getElementById("stepval").innerHTML = steps.value;
            promptdc.send(JSON.stringify({steps: steps.value}))
        }
        seed.oninput = function() {
            promptdc.send(JSON.stringify({seed: seed.value}))
        }
        reference.oninput = function() {
            promptdc.send(JSON.stringify({ref: reference.checked}))
        }

        set_reference.onclick = function() {
            promptdc.send(JSON.stringify({set_ref: true}))
        }

        style_fidelity.oninput = function() {
            promptdc.send(JSON.stringify({style_fidelity: style_fidelity.value}))
        }

        controlnet.oninput = function() {
            promptdc.send(JSON.stringify({controlnet: controlnet.checked, ref: false}))
            if (!controlnet.checked) {
                reference.checked = false
            }
            
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
        .getUserMedia({ video: {facingMode: 'user'}, audio: true })
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
