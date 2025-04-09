import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

// const enableWebcamButton = document.getElementById("webcamButton");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const displayOutputText = document.getElementById("predictionVal")

const drawUtils = new DrawingUtils(canvasCtx);
let poseLandmarker = undefined;
let webcamRunning = false;
let results = undefined;
let nn

function createNeuralNetwork() {
    ml5.setBackend('webgl')
    nn = ml5.neuralNetwork({task: 'classification', debug: true})
    const options = {
        model:"model/model.json",
        metadata:"model/model_data.json",
        weights:"model/model.weights.bin",
    }
    nn.load(options, createPoseLandmarker())
}


/********************************************************************
 // CREATE THE POSE DETECTOR
 ********************************************************************/
const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1
    });
    console.log("Pose model loaded, you can start webcam");


    enableCam()
    // logButton.addEventListener("click", (e) => classifyPose());
};

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
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            videoView.style.height = video.videoHeight + "px";
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
    results = await poseLandmarker.detectForVideo(video, performance.now());

    if (results.landmarks.length > 0) {
        const pose = results.landmarks[0];
        const nose = pose[0]; // landmark index 0 is usually the nose

        classifyPose()
    }

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    for (let pose of results.landmarks) {
        drawUtils.drawConnectors(pose, PoseLandmarker.POSE_CONNECTIONS, { color: "#00FF00", lineWidth: 4 });
        drawUtils.drawLandmarks(pose, { radius: 3, color: "#FF0000", lineWidth: 2 });
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

function classifyPose(){
    displayOutputText.innerText = "classifying"
    // let numbersonly = []
    // let hand = results.landmarks[0]
    // for (let point of hand) {
    //     numbersonly.push(point.x,point.y,point.z)
    // }
    // console.log(numbersonly)
    // nn.classify(numbersonly, (results) => {
    //     console.log(results)
    // })
}

/********************************************************************
 // LOG POSE COORDINATES IN THE CONSOLE
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
//                     logAllPoses(label)
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
// function logAllPoses(label) {
//     let data = []
//     for (let pose of results.landmarks[0]) {
//         // console.log(`Label: ${JSON.stringify(label)}`)
//         // console.log(`Here is my object: ${JSON.stringify(pose[0], null, 2)}`);// Example: nose
//         // You can log others like pose[11] (left shoulder), pose[23] (left hip), etc.
//
//         data.push(pose.x, pose.y, pose.z)
//     }
//     allData.push({points: data, label: label})
//     console.log(allData)
//     localStorage.setItem("poseData", JSON.stringify(allData))
// }



/********************************************************************
 // START THE APP
 ********************************************************************/
if (navigator.mediaDevices?.getUserMedia) {
    createPoseLandmarker();
}

createNeuralNetwork()
