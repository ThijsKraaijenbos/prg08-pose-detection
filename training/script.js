import posedata from './data.json' with {type: "json"}
let nn

function startTraining() {
    ml5.setBackend('webgl')
    nn = ml5.neuralNetwork({task: 'classification', debug: true})
    console.log(nn)

    for(let pose of posedata) {
        // console.log(pose)
        nn.addData(pose.points, {label: pose.label})
    }

    nn.normalizeData()
    nn.train({epochs:75}, finishedTraining)
}

function finishedTraining() {
    console.log("finished training!")
    let demopose = posedata[10].points
    nn.classify(demopose, (results) => {
        console.log(`i think this is a ${results[0].label}`)
        console.log(`${(results[0].confidence.toFixed(2)) * 100}% sure`)
    })
    nn.save()
}

startTraining()