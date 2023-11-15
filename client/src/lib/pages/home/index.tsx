'use client';

import { Flex, Box, AspectRatio, IconButton, ButtonGroup, SimpleGrid, GridItem, FormControl, FormLabel, Input, Textarea, RangeSlider, VStack, Select, Button, Spacer } from '@chakra-ui/react';
import { Html } from 'next/document';
import { userAgentFromString } from 'next/server';
import React, { useState, useRef, useEffect, useCallback, useMemo, use, ChangeEvent } from 'react';
import { FullScreen, useFullScreenHandle } from "react-full-screen";
import { useOrientation } from "@uidotdev/usehooks";
import {
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  SliderMark,
  Switch,
} from '@chakra-ui/react'

import { Icon } from '@chakra-ui/react'
import { MdFlipCameraIos } from 'react-icons/md'
import { FaDice} from 'react-icons/fa'

import SliderParameter from '~/lib/components/SliderParameter';

const pc_config = {
	iceServers: [
		{
			'urls': 'turn:blendotron.art:51820',
      'username': 'videosd',
      'credential': 'videosd',
		},
	],
};

let initOptions = {
  "prompt": "Underwater",
  "strength": 0.4,
  "guidance_scale": 5,
  "steps": 2,
  "seed": 23,
  "ref": false,
  "style_fidelity": 1,
  "controlnet": true, 
  "width": 512,
  "height": 512,
};

const promptExamples = [
  "Anime",
  "Star wars",
  "Pencil drawing",
  "Underwater",
  "Pixar cartoon, cg",
  "On fire",
  "White marble statues"
]

const Home = () => {
  const pcRef = useRef<RTCPeerConnection>();
  const dcRef = useRef<RTCDataChannel>();
  let facingMode = 'user';
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);
  let isConnecting = false;
  // store options as object
  

  let [options, setOptions] = useState(initOptions);
	const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);
	const localStreamRef = useRef<MediaStream>();
  const senderRef = useRef<RTCRtpSender>();
  const orientation = useOrientation();
  let called = 0;

  const getLocalStream = async () => {
    called++;
    console.log(`get localstream ${called}`, facingMode)
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => {
        track.stop();
      });
    }
		try {
			const localStream = await navigator.mediaDevices.getUserMedia({
				audio: false,
				video: { 
          facingMode: facingMode,
				},
			});
      // detect if user has multiple cameras
      
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter((device) => device.kind === 'videoinput');
      if (videoDevices.length > 1) {
        console.log("multiple cameras detected");
        setHasMultipleCameras(true);
        // add flip camera switch
      }

			localStreamRef.current = localStream;
			if (localVideoRef.current) localVideoRef.current.srcObject = localStream;
      localStream.getTracks().forEach((track) => {
        if (senderRef.current) senderRef.current.replaceTrack(track);
      });
		} catch (e) {
			console.log(`getUserMedia error: ${e}`);
		}
	};

  const createConnection = useCallback(() =>{
		try {
			const pc = new RTCPeerConnection(pc_config);
      pcRef.current = pc;

			pc.onicecandidate = (e) => {
				if (!(e.candidate)) return;
				console.log('onicecandidate');
			};

			pc.oniceconnectionstatechange = (e) => {
				console.log(e);
			};

			pc.ontrack = (e) => {
				console.log('ontrack success');
        if (remoteVideoRef.current) remoteVideoRef.current.srcObject = e.streams[0];
			};

			if (localStreamRef.current) {
				console.log('localstream add');
				senderRef.current = pc.addTrack(localStreamRef.current.getTracks()[0], localStreamRef.current);
			} else {
				console.log('no local stream');
			}

      dcRef.current = pc.createDataChannel('prompt', {"ordered": true});
      dcRef.current.onmessage = (e) => {
        console.log(e.data);
      }

			return pc;
		} catch (e) {
			console.error(e);
			return undefined;
		}
	}, [localStreamRef.current]);

  const [isStreaming, setIsStreaming] = useState(false);
  
  const negotiate = useCallback(async () => {
    pcRef.current = createConnection();
    const pc = pcRef.current;
    if (!pc) return;
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const answer = await fetch('https://blendotron.art/offer', {
        body: JSON.stringify({
            sdp: offer.sdp,
            type: offer.type,
            options: initOptions
        }),
        headers: {
            'Content-Type': 'application/json'
        },
        method: 'POST'
    }).then((a) => a.json());
    
    pc.setRemoteDescription(answer);

    console.log(answer);
  }, [localStreamRef.current]);

  const handle = useFullScreenHandle();

  const updateInitOptions = () => {
    if (!remoteVideoRef.current) return;
    const dwidth = remoteVideoRef.current.offsetWidth;
    const dheight = remoteVideoRef.current.offsetHeight;

    // find aspect ratio of video container
    const dar = dwidth / dheight;
    // set 512x512 if aspect ratio is 1:1
    if (dar == 1) {
      initOptions.width = 768;
      initOptions.height = 768;
    } else if (dar > 1) {
      initOptions.width = 768;
      initOptions.height = 768 / dar;
    } else {
      initOptions.width = 768 * dar;
      initOptions.height = 768;
    }
              // round to nearest multiple of 16
    initOptions.width = Math.round(initOptions.width / 16) * 16;
    initOptions.height = Math.round(initOptions.height / 16) * 16;

    console.log(remoteVideoRef.current?.offsetWidth, remoteVideoRef.current?.offsetHeight);

  }

  useEffect(() => {
    console.log(isConnecting)
    if (!isConnecting) {
      isConnecting = true;
      console.log("Negotiating...");
      getLocalStream().then(() => {
        if (remoteVideoRef.current) {
          updateInitOptions();
          negotiate().then(() => { 
            setIsStreaming(true);
          });
        }
      });
    }
  }, [facingMode]);

  const orientationChange = useCallback(() => {
    console.log(orientation);
    return orientation;
  }, [orientation]);

  useEffect(() => {
    console.log(orientationChange())
  }, [orientationChange]);

  const handleChange = (name: string, value: any) => {
    setOptions(prevState => ({
        ...prevState,
        [name]: value
    }));
        console.log(options);
        if (!dcRef.current) return;
        if (dcRef.current?.readyState == "open")
          console.log("Sending in data channel..")
          dcRef.current.send(JSON.stringify({[name]: value}));
  };

  const flipCamera = (e: ChangeEvent) => {
    if (facingMode == 'environment') {
      facingMode = 'user';
    } else {
      facingMode = 'environment';
    }
    console.log("flping camera", facingMode)
    getLocalStream();
  };
  
  const portraitRatio = 9/12

  return (
    <FullScreen handle={handle}>
      <Flex
        direction={["column","row","row"]}
        gap={[3,2]}
        align={["center","flex-start"]}
        maxWidth={["full", "full", "full"]}
        minH='full'
        h="full"
      >
          <AspectRatio minW={['full', '79vw']} maxH={["80vh","90vh"]} ratio={[portraitRatio,16/9]}>
            <video ref={remoteVideoRef} autoPlay playsInline />
          </AspectRatio>
          <SimpleGrid columns={4} spacing={3} minH={0} minW={['full', '20vw', '20vw']}  ml={[2,0]} p={[1,2]}>
            <GridItem order={[1,0]} colSpan={[1,4]}>
              <AspectRatio ratio={[portraitRatio,16/9]}>
                <video ref={localVideoRef} autoPlay playsInline />
              </AspectRatio>
              
              {hasMultipleCameras && (
                <FormControl display="flex" alignItems="center">
                  <FormLabel htmlFor="cameras" mb="0" fontSize={['3xs','inherit']}>
                    Flip Camera
                  </FormLabel>
                  <Switch id='cameras' size={['sm','md']} py={2} isDisabled={!hasMultipleCameras} onChange={flipCamera}/>

                </FormControl>
              )}
            </GridItem>

            <GridItem colSpan={[3,4]} order={[0,1]}>
              <FormControl>
                <VStack w="full" align="left" spacing={[1,2]}>
                  <Select size={['xs','sm']} placeholder="Example prompts..." isDisabled={!isStreaming} onChange={(val) => handleChange("prompt", val.target.value)}>
                    {promptExamples.map((prompt) => (
                      <option key={prompt} value={prompt}>{prompt}</option>
                    ))}
                  </Select>


                  <Box mb={1}>
                    <Textarea isDisabled={!isStreaming} fontSize={["2xs","initial"]} p={[2,2]} placeholder="Type your prompt here..." onChange={(val) => handleChange("prompt", val.target.value)} value={options.prompt} /> 
                  </Box>

                
                  <SliderParameter label="steps" isDisabled={!isStreaming} min={1} max={12} step={1} defaultValue={options.steps} onChange={(val) => handleChange("steps", val)}>
                    Steps: {options.steps}
                  </SliderParameter>

                  <SliderParameter label="CFG" isDisabled={!isStreaming} min={1} max={20} step={0.25} defaultValue={options.guidance_scale} onChange={(val) => handleChange("guidance_scale", val)}>
                    CFG: {options.guidance_scale}
                  </SliderParameter>

                  <SliderParameter label="strength" isDisabled={!isStreaming} min={0} max={1} step={0.02} defaultValue={options.strength} onChange={(val) => handleChange("strength", val)}>
                    Strength: {options.strength}
                  </SliderParameter>

                  <Flex w="full" my={[1,2]} direction={"row"} alignContent={"flex-start"} alignItems={"center"}>
                  <FormLabel fontSize={['2xs','inherit']} mb={0} mr={1}>Seed:</FormLabel>
                  
                  <ButtonGroup size='sm' isAttached variant='outline'>
                    
                    <Input size={['xs','sm']} width={32} placeholder="Seed" isDisabled={!isStreaming} onChange={(val) => handleChange("seed", val.target.value) } value={options.seed}/>
                    
                    <Button ml={2} aria-label="Randomize" size={['xs','sm']} isDisabled={!isStreaming} onClick={() => handleChange("seed", Math.floor(Math.random() * 100000000))} leftIcon={<FaDice/>}> 
                      Randomize
                    </Button>
                  </ButtonGroup>
                  </Flex>
                </VStack>
              </FormControl>
            </GridItem>


          </SimpleGrid>
      </Flex>
      

    </FullScreen>

  );
};

export default Home;
