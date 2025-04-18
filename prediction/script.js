import {
    DrawingUtils,
    FilesetResolver,
    HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

// const enableWebcamButton = document.getElementById("webcamButton");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const displayOutputText = document.getElementById("predictionVal")

const controlsTag = document.getElementById('controlsVal');
let controlsToggled = true

const audioTag = document.getElementById("customAudio")
const audioTitle = document.getElementById("audioTitle")
const seekbar = document.getElementById('seekbar');
const durationElmt = document.getElementById('duration');
const volumeTag = document.getElementById('volume');
const playlist = document.getElementById("playlist")
let lastUpdateType = ""
let updateCounter = 0

let songIndex = 0

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

    loadPlayList()
    enableCam()
}

/********************************************************************
// START THE WEBCAM
********************************************************************/
async function enableCam() {
    const videoView = document.getElementById("videoView")
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
            video.style.height = video.videoHeight + "px";
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
        updateCounter = 0
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
        //needs high confidence to work
        if (results[0].confidence < 0.98) {
            return
        }
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
                displayOutputText.innerText = "Toggling inputs"
                break;
        }

        updateAudioTag(results[0].label)
    })
}

function updateAudioTag(updateType) {
    //Check for 50 frames if the prediction is the same
    //This fixes individual frames accidentally randomly causing updates when the algorithm is being goofy
    console.log(updateType)
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
            controlsToggled = !controlsToggled
            controlsToggled ? controlsTag.innerText = `Inputs are turned on` : controlsTag.innerText = `Inputs are turned off`
        }
        return;
    }

    if (!controlsToggled) {
        return
    }

    switch (updateType) {
        case "Play":
            if (!audioTag.src || audioTag.src === window.location.href) {
                const firstElmt = document.getElementById("songIndex-0");
                if (firstElmt) {
                    const parentDiv = firstElmt.parentElement

                    audioTag.src = firstElmt.dataset.filepath;
                    audioTitle.innerText = firstElmt.innerText;
                    audioTag.dataset.songIndex = "0";
                    parentDiv.classList.add("bg-blue-400")
                    audioTag.play();
                }
            } else {
                audioTag.play();
            }
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
    // Set the seekbar's max value to the audio's duration
    seekbar.max = audioTag.duration;
});

audioTag.addEventListener('timeupdate', () => {
    seekbar.value = audioTag.currentTime;  // Sync the seekbar with audio progress
    durationElmt.textContent = formatTime(audioTag.currentTime);

    //Go to the next song in the playlist if current song is finished
    if (audioTag.currentTime === audioTag.duration) {
        document.querySelector(".bg-blue-400").classList.remove("bg-blue-400")

        const currentSongIndex = parseInt(audioTag.dataset.songIndex.split("-")[1]);
        const nextItem = document.getElementById(`songIndex-${currentSongIndex + 1}`)
        audioTag.dataset.songIndex = `songIndex-${currentSongIndex + 1}`;

        if (nextItem) {
            const parentDiv = nextItem.parentElement

            audioTag.src = nextItem.dataset.filepath
            audioTitle.innerText = nextItem.innerText
            parentDiv.classList.add("bg-blue-400")
            audioTag.play()
        } else {
            audioTag.src = ""
            audioTitle.innerText = "You've finished your playlist."
        }
    }
});

function formatTime(time) {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60).toString().padStart(2, '0');
    return `${minutes}:${seconds}`;
}


async function loadPlayList() {
    //add loading
    const loading = document.createElement("h1")
    loading.innerText = "loading playlist..."
    playlist.appendChild(loading)

    try {
        const response = await fetch(`http://localhost:8000/api/playlist`, {
            method: 'GET',
        });

        const songData = await response.json();
        console.log(songData)

        playlist.addEventListener("click", function (e) {
            if (e.target.tagName === "H1" || e.target.classList.contains("song-item")) {
                const prevItem = document.querySelector(".bg-blue-400")
                if (prevItem) {
                    prevItem.classList.remove("bg-blue-400")
                }

                const parentDiv = e.target.closest(".song-item")
                const h1 = parentDiv.children[0]

                audioTag.src = h1.dataset.filepath
                audioTitle.innerText = h1.innerText
                audioTag.dataset.songIndex = h1.id
                parentDiv.classList.add("bg-blue-400")
                audioTag.play()
            }
        })

        if (songData !== "") {
            for (let song of songData) {
                new PlaylistElement(song)
            }
        }

        const playlistInput = document.getElementById("playlistInput")
        playlistInput.addEventListener("change", uploadSong)
        playlistInput.classList.remove("hidden")
    } catch (e) {
        console.log(e)
    } finally {
        //remove loading
        playlist.children[0].remove()
    }
}

async function uploadSong() {
    const files = document.getElementById("playlistInput").files

    for (let file of files) {
        const formData = new FormData();
        formData.append('playlistInput', file);

        try {
            const response = await fetch(`http://localhost:8000/api/playlist`, {
                method: 'POST',
                body: formData
            });

            const songData = await response.json();
            console.log(songData)
            new PlaylistElement(songData)

        } catch (e) {
            console.log(e)
        }
    }
}

function PlaylistElement(songData) {
    const div = document.createElement("div")
    div.className = "song-item relative w-64 bg-gray-700 rounded-xl p-2 min-h-16 max-h-16 flex items-center justify-center shadow-lg border border-gray-700 space-y-4"

    const songItem = document.createElement("h1")
    songItem.id = `songIndex-${songIndex}`
    songItem.dataset.filepath = songData.file_path
    songItem.innerText = songData.name

    const removeButton = document.createElement("p")
    removeButton.innerText = "X"
    removeButton.className = "text-red-500 text-bold absolute right-1 top-0 !m-0"
    removeButton.addEventListener("click", async function (e) {
        try {
            const result = await fetch(`http://localhost:8000/api/playlist/${songData.id}`, {
                method: 'delete',
            });
            if (result) {
                e.target.parentElement.remove()
            }

        } catch (e) {
            console.log(e)
        }
    })

    div.appendChild(songItem)
    div.appendChild(removeButton)
    playlist.appendChild(div)
    songIndex++
}

createNeuralNetwork()
