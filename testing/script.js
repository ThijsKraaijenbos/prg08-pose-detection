import testdata from './testData.json' with {type: "json"}
let nn


function createNeuralNetwork() {
    ml5.setBackend('webgl')
    nn = ml5.neuralNetwork({task: 'classification', debug: true})
    const options = {
        model: "./model/model.json",
        metadata: "./model/model_meta.json",
        weights: "./model/model.weights.bin",
    }
    nn.load(options, classifyTestData)
}

function classifyTestData() {
    console.log(testdata)
    for (let pose of testdata) {
        nn.classify(pose.points, (results) => {
            // console.log(results)
            console.log(`i think this is a ${results[0].label}. It actually is a ${pose.label}`)
            // console.log(`${(results[0].confidence.toFixed(2)) * 100}% sure`)
        })
    }
}

createNeuralNetwork()