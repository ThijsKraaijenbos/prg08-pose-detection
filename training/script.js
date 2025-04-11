import posedata from './data.json' with {type: "json"}
let nn

function startTraining() {
    ml5.setBackend('webgl')
    const options = {
        task: 'classification',
        debug: true,
        layers: [
            {
                type: 'dense',
                units: 64,
                activation: 'relu',
            }, {
                type: 'dense',
                units: 64,
                activation: 'relu',
            },
            {
                type: 'dense',
                activation: 'softmax',
            },
        ]
    }
    nn = ml5.neuralNetwork(options)
    console.log(nn)

    for(let pose of posedata) {
        // console.log(pose)
        nn.addData(pose.points, {label: pose.label})
    }

    nn.normalizeData()
    nn.train({epochs:100}, finishedTraining)
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