'use client';

import { Flex, Box, AspectRatio, Tabs, Tab, TabList, TabPanel, TabPanels, IconButton, ButtonGroup, SimpleGrid, GridItem, FormControl, FormLabel, Input, Textarea, RangeSlider, VStack, Select, Button, Spacer, Skeleton, AbsoluteCenter, useDisclosure, Spinner, Fade, useToast } from '@chakra-ui/react';
import { Html } from 'next/document';
import { userAgentFromString } from 'next/server';
import React, { useState, useRef, useEffect, useCallback, ChangeEvent } from 'react';
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
import { FaRegCirclePlay } from "react-icons/fa6";
import { IoPlaySharp, IoStopSharp } from "react-icons/io5";


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
  "prompt": "Pixar cartoon, cg",
  "strength": 0.4,
  "guidance_scale": 5,
  "steps": 4,
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
  const [facingMode, setFacingMode] = useState('user');
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  
  const [previewOpen, setPreviewOpen] = useState(false);
  

  let [options, setOptions] = useState(initOptions);
	const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);
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
      if (senderRef.current) {
        senderRef.current.replaceTrack(localStream.getVideoTracks()[0]);
      }
		} catch (e: any) {
      e.name == "NotAllowedError" ? console.log("User denied camera access") :
			console.log(e);
      throw e;
		}
	}

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
  
  const negotiate = async () => {
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
  };

  const [isFull, setIsFull] = useState(false);
  const fsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!fsRef.current) return;
    if (isFull) {
      fsRef.current.requestFullscreen().then(() => {
        setWindowDimensions();
      })
    } else if (document.fullscreenElement != null) {
      document.exitFullscreen().then(() => {
        setWindowDimensions();
      })
    }
  }, [isFull]);

  const handleFullscreenChange = useCallback(() => {
    document.fullscreenElement != null ? setIsFull(true) : setIsFull(false);
  }, []);

  useEffect(() => {
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    }
  },[]);


  const updateInitOptions = () => {
    if (!videoContainerRef.current || !remoteVideoRef.current) return;
    const dwidth = videoContainerRef.current.offsetWidth;
    const dheight = videoContainerRef.current.offsetHeight;


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
    
    // remoteVideoRef.current.style.height = `${initOptions.height}px`;
    // remoteVideoRef.current.style.width = `${initOptions.width}px`;
    // videoContainerRef.current.style.width = `${dwidth}px`;

    console.log(initOptions.width, initOptions.height);

  }

  const [startVideo, setStartVideo] = useState(false);

  useEffect(() => {
    console.log(isConnecting)
    if (!isConnecting && startVideo) {
      setIsConnecting(true);
      console.log("Negotiating...");
      getLocalStream().then(() => {
        if (remoteVideoRef.current) {
          updateInitOptions();
          negotiate().then(() => { 
            console.log("negotiated");
            setIsConnecting(false);
            setIsStreaming(true);
            setPreviewOpen(true);
          });
        } else {
          console.log("remote video ref is null");
          setIsConnecting(false);
        }
      }).catch((e) => {
        console.log(e);
        setIsConnecting(false);
      });
    }
    if (isStreaming && startVideo == false) {
      console.log("Stopping stream");
      setIsStreaming(false);
      setIsConnecting(false);
      setPreviewOpen(false)
      if (localStreamRef.current) {
        pcRef.current?.close();
      }
    }
  }, [startVideo, facingMode]);

  const handleChange = useCallback((name: string, value: any) => {
    setOptions(prevState => ({
        ...prevState,
        [name]: value
    }));
        console.log(name, value)
        console.log(options);
        if (!dcRef.current) return;
        if (dcRef.current?.readyState === "open")
          console.log("Sending in data channel..")
          dcRef.current.send(JSON.stringify({[name]: value}));
  }, [options]);

  useEffect(() => {
    const cleanup = () => {
      if (!pcRef.current) return;
      pcRef.current?.getSenders().forEach((sender) => {
        sender.track?.stop();
      });
      pcRef.current?.close();
    }
  
    window.addEventListener('beforeunload', cleanup);
  
    return () => {
      window.removeEventListener('beforeunload', cleanup);
    }
  }, []);

  const flipCamera = (e: ChangeEvent) => {
    if (facingMode == 'environment') {
      setFacingMode('user');
    } else {
      setFacingMode('environment');
    }
    
    console.log("flipping camera", facingMode)
    getLocalStream();
  };

  const setWindowDimensions = useCallback(() => {
    if (!videoContainerRef.current) return;

    
    handleChange("width", videoContainerRef.current.offsetWidth);
    handleChange("height", videoContainerRef.current.offsetHeight);
  },[handleChange])

  useEffect(() => {
    window.addEventListener('resize', setWindowDimensions);
    return () => {
      window.removeEventListener('resize', setWindowDimensions)
    }
  }, [])

  const { isOpen, onToggle, onOpen, onClose } = useDisclosure()
  const toast = useToast()  

  return (


    <Box flex={1} h={'auto'} ref={fsRef}>
      

      <Flex
        direction={["column","row"]}
        gap={[0,0]}
        h={'full'}
        w={'full'}
        alignItems={["center","stretch"]}
        alignContent={["center","stretch"]}
      >   
            <Box ref={videoContainerRef} flex={'auto'} maxH={['90vh', '100vh']} width={['full','auto']} h={'100%'} bgColor={'black'} position='relative' onClick={isStreaming ? onToggle : ()=>{}}>
                <Box visibility={isStreaming ? 'visible': 'hidden'} height={['initial','auto', 'auto']}  flex={1} >
                  <video style={{width: "100%", height: "100%"}} ref={remoteVideoRef} autoPlay playsInline />
                </Box>
                <AbsoluteCenter>
                  {isConnecting && (
                    <Spinner size="md" color="white" />
                  )}
                  {!isStreaming && !isConnecting && (
                    <IconButton
                    isRound={true}
                    variant='solid'
                    colorScheme='red'
                    aria-label='Done'
                    fontSize='20px'
                    pl={'4px'}
                    icon={<IoPlaySharp/>}
                    onClick={async () => {
                      // Create an example promise that resolves in 5s
                      const permissionObj = await navigator.permissions.query({name: 'camera' as PermissionName});

                      console.log(permissionObj)
                      if (permissionObj.state === 'granted') {
                        setStartVideo(true);
                        //setStartVideo(true);
                        //permission has already been granted, no prompt is shown
                      } else if (permissionObj.state === 'prompt' || permissionObj.state === 'denied') {
                        const getCameraPermissionsPromise = new Promise((resolve, reject) => {
                              permissionObj.onchange = (e) => {
                                if (!e.target) resolve("error");
                                const permissionStatus = e.target as PermissionStatus;
                                if (!permissionStatus.state) resolve("denied");
                                if (permissionStatus.state == "granted") {
                                  setStartVideo(true);
                                  resolve(permissionObj.state);
                                } else {
                                  setStartVideo(false);
                                  reject("denied");
                                }
                              }
                              getLocalStream().then(() => {
                                setStartVideo(true);
                                console.log("prompting for camera access")
                              }).catch((e) => {
                                console.log(e);
                                reject("denied");
                              })
                              //there's no peristent permission registered, will be showing the prompt
                            
                            console.log(permissionObj.state);
                          })
                        toast.promise(getCameraPermissionsPromise, {
                          success: { title: 'Camera access acquired!', description: 'Good to go!', duration: 1000 },
                          error: { title: 'Camera Access required!', description: "Camera access is required to use this app. We will not store or record your video feed in any way. You can reset the camera permission for this site in your browser's settings." },
                          loading: { title: 'Plese allow us to use your camera.', description: 'Your video stream will not be recorded.' },
                        })
                          
                
                      // Will display the loading toast until the promise is either resolved
                      // or rejected.

                    }}}
                    
                  />
                  )}
                </AbsoluteCenter>
                <Fade in={isOpen} onHoverStart={onOpen} onHoverEnd={onClose}>
                  <Box
                    p={2}
                    color='white'
                    opacity={isOpen? 0.6 : 0}
                    bottom={0}
                    right={0}
                    width={'100%'}
                    textAlign={'center'}
                    shadow='md'
                    position='absolute'
                  >
                    {isStreaming && (
                      <IconButton
                      isRound={true}
                      variant='solid'
                      opacity={isOpen? 1 : 0}
                      colorScheme='red'
                      aria-label='Done'
                      bg='red.300'
                      fontSize='20px'
                      icon={<IoStopSharp/>}
                      onClick={() => setStartVideo(false)}
                    />
                    )}
                  </Box>
                </Fade> 
            </Box>

          

          <Box visibility={previewOpen ? 'visible' : 'hidden'} position={'absolute'} boxShadow='dark-lg' w={['20vw', '10vw']} top={0} left={0} m={2}>
            <video  ref={localVideoRef} autoPlay playsInline />
          </Box>



              <Tabs flex={'initial'} size={['sm','md']} minH={160} isFitted w={['full','auto']}>
                <TabList>
                  <Tab>Prompt</Tab>
                  <Tab>Diffusion</Tab>
                  <Tab>Settings</Tab>
                </TabList>
                <TabPanels>
                  <TabPanel>
                    <Select size={['xs','sm']} placeholder="Example prompts..." isDisabled={!isStreaming} onChange={(val) => handleChange("prompt", val.target.value)}>
                      {promptExamples.map((prompt) => (
                        <option key={prompt} value={prompt}>{prompt}</option>
                      ))}
                    </Select>

                    <Box mt={1} >
                      <Textarea minH={16} fontSize={'sm'} isDisabled={!isStreaming} p={[2,2]} placeholder="Type your prompt here..." onChange={(val) => handleChange("prompt", val.target.value)} value={options.prompt} /> 
                    </Box>
                  </TabPanel>
                  <TabPanel>
                  <SliderParameter label="steps" isDisabled={!isStreaming} min={1} max={12} step={1} defaultValue={options.steps} onChange={(val) => handleChange("steps", val)}>
                    Steps: {options.steps}
                  </SliderParameter>

                  <SliderParameter label="strength" isDisabled={!isStreaming} min={0} max={1} step={0.02} defaultValue={options.strength} onChange={(val) => handleChange("strength", val)}>
                    Strength: {options.strength}
                  </SliderParameter>
                  <Flex mt={4} direction="row" align="center" justify="left" width="full">
                    <FormLabel fontSize={['xs','initial']} mb={0}>Seed:</FormLabel>
                    
                    <Input w={28} size={['xs','sm']} placeholder="Seed" isDisabled={!isStreaming} onChange={(val) => handleChange("seed", val.target.value) } value={options.seed}/>
                    
                    <IconButton ml={1} aria-label="Randomize" size={['xs','sm']} isDisabled={!isStreaming} onClick={() => handleChange("seed", Math.floor(Math.random() * 100000000))} icon={<FaDice/>}/>
                  </Flex>
                  </TabPanel>
                  <TabPanel>
                    <FormControl display="flex" alignItems="center">
                        <Switch id='preview' isChecked={previewOpen} mr={2} py={1.5} onChange={(e) => setPreviewOpen(e.target.checked)}/>

                        <FormLabel htmlFor="preview" mb="0">
                          Camera Preview
                        </FormLabel>
                    </FormControl>

                    <FormControl display="flex" alignItems="center">
                        <Switch id='fullscreen' isChecked={isFull} mr={2} py={1.5} onChange={(e) => setIsFull(e.target.checked)} />

                        <FormLabel htmlFor="fullscreen" mb="0">
                          Fullscreen
                        </FormLabel>
                    </FormControl>
                    
                    {hasMultipleCameras && (
                      <FormControl display="flex" alignItems="center">
                        <Switch id='cameras' py={1.5} mr={2} isDisabled={!hasMultipleCameras} onChange={flipCamera}/>

                        <FormLabel htmlFor="cameras" mb="0">
                          Flip Camera
                        </FormLabel>
                      </FormControl>
                    )}
                  </TabPanel>
                </TabPanels>
              </Tabs>

      </Flex>
    {/*  */}
    </Box>
  );
};

export default Home;
