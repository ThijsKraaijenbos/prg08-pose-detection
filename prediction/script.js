import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

// const enableWebcamButton = document.getElementById("webcamButton");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const displayOutputText = document.getElementById("predictionVal")

const controlsTag = document.getElementById('controlsVal');
let inputControls = true

const audioTag = document.getElementById("customAudio")
const seekbar = document.getElementById('seekbar');
const durationElmt = document.getElementById('duration');
const volumeTag = document.getElementById('volume');
let lastUpdateType = ""
let updateCounter = 0

const drawUtils = new DrawingUtils(canvasCtx);
let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;
let nn

function createNeuralNetwork() {
    ml5.setBackend('webgl')
    nn = ml5.neuralNetwork({task: 'classification', debug: true})
    const options = {
        model: "./model/model.json",
        metadata: "./model/model_meta.json",
        weights: "./model/model.weights.bin",
    }
    nn.load(options, createHandLandmarker())
}

/********************************************************************
// CREATE THE POSE DETECTOR
********************************************************************/

const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });
    console.log("model loaded, you can start webcam")

    enableCam()
}

/********************************************************************
// START THE WEBCAM
********************************************************************/
async function enableCam() {
    const videoView = document.querySelector(".videoView")
    const loadingElmt = document.createElement("p")
    loadingElmt.innerText = "Loading webcam..."
    videoView.appendChild(loadingElmt)

    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam();
            loadingElmt.remove()
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

/********************************************************************
// START PREDICTIONS    
********************************************************************/
async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now())

    let hand = results.landmarks[0]
    if(hand) {
        classifyPose()
    } else if (displayOutputText.innerText !== "No input detected") {
        displayOutputText.innerText = "No input detected"
    }

    // canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    // for(let hand of results.landmarks){
    //     drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
    //     drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });
    // }

    if (webcamRunning) {
       window.requestAnimationFrame(predictWebcam)
    }
}

function classifyPose(){

    let numbersonly = []
    let hand = results.landmarks[0]

    for (let point of hand) {
        numbersonly.push(point.x,point.y,point.z)
    }

    nn.classify(numbersonly, (results) => {

        switch (results[0].label) {
            case "Play":
                displayOutputText.innerText = "Playing music"
                break;
            case "Pause":
                displayOutputText.innerText = "Pausing music"
                break;
            case "VolumeUp":
                displayOutputText.innerText = "Raising volume"
                break;
            case "VolumeDown":
                displayOutputText.innerText = "Lowering volume"
                break;
            case "ToggleControls":
                displayOutputText.innerText = "Toggling camera input controls"
                break;
        }

        updateAudioTag(results[0].label)
    })
}

function updateAudioTag(updateType) {
    //Check for 50 frames if the prediction is the same
    //This fixes individual frames accidentally randomly causing updates when the algorithm is being goofy
    if (updateType !== lastUpdateType) {
        updateCounter = 0
        lastUpdateType = updateType
        return
    }
    if (updateCounter < 30) {
        updateCounter++
        lastUpdateType = updateType
        return;
    }

    if (updateType === "ToggleControls") {
        updateCounter++
        //Check for 1 extra frame and then lock the toggle based on the fact that it only
        //fires on the 31st frame, so it doesn't toggle every frame.
        //(locking on the 30th frame doesn't work because of the if statement earlier in the function)
        if (updateCounter === 31) {
            inputControls = !inputControls
            inputControls ? controlsTag.innerText = `Inputs are turned on` : controlsTag.innerText = `Inputs are turned off`
        }
        return;
    }


    switch (updateType) {
        case "Play":
            audioTag.play()
            break;

        case "Pause":
            audioTag.pause()
            break;

        case "VolumeUp":
            volumeTag.innerText = `Volume: ${(audioTag.volume * 100).toFixed(0)}%`
            if (audioTag.volume > 0.997) {
                audioTag.volume = 1
                return;
            }
            audioTag.volume += 0.003 //changeamount can be negative, this will just add the negative amount which lowers it
            break;

        case "VolumeDown":
            volumeTag.innerText = `Volume: ${(audioTag.volume * 100).toFixed(0)}%`
            if (audioTag.volume < 0.003) {
                audioTag.volume = 0
                return;
            }
            audioTag.volume -= 0.003 //changeamount can be negative, this will just add the negative amount which lowers it
            break;
    }
}

audioTag.addEventListener('loadedmetadata', () => {
    seekbar.max = audioTag.duration;
    console.log("Audio max =" + audioTag.duration)
});

audioTag.addEventListener('timeupdate', () => {
    seekbar.value = audioTag.currentTime;
    durationElmt.textContent = formatTime(audioTag.currentTime);
});
function formatTime(time) {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60).toString().padStart(2, '0');
    return `${minutes}:${seconds}`;
}

/********************************************************************
// LOG HAND COORDINATES IN THE CONSOLE
********************************************************************/
// function startCountdown(seconds) {
//     logging = true
//     let counter = seconds;
//     let loggedPoseCount = 0
//     let label = document.getElementById("labelInput").value
//
//     if (logging === true) {
//         const interval = setInterval(() => {
//             console.log(counter);
//             counter--;
//
//             if (counter < 0 ) {
//                 clearInterval(interval);
//
//                 const logInterval = setInterval(() => {
//                     logAllHands(label)
//                     loggedPoseCount++
//                     if (loggedPoseCount === 50) {
//                         clearInterval(logInterval)
//                         console.log("Finished Logging")
//                     }
//                 }, 100)
//
//
//                 logging = false
//                 console.log("Logging is now stopped")
//             }
//         }, 1000);
//     }
// }
// function logAllHands(label) {
//     let data = []
//     for (let pose of results.landmarks[0]) {
//         data.push(pose.x, pose.y, pose.z)
//     }
//     allData.push({points: data, label: label})
//     console.log(allData)
//     localStorage.setItem("handData", JSON.stringify(allData))
// }

/********************************************************************
// START THE APP
********************************************************************/
// if (navigator.mediaDevices?.getUserMedia) {
//     createHandLandmarker()
// }





createNeuralNetwork()
