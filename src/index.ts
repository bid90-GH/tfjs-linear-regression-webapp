import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

import {showModelSummary, showLayerModel, showFitCallbacks, scatterPlot} from './views'
import {normalizeTensor, denormalizeTensor} from './utils'

let points: { x: string, y: string }[]
let model
let featureTensor
let labelTensor
let featureTensorNormalized
let labelTensorNormalized
let trainingFeatures
let testingFeatures
let trainingLabel
let testingLabel

const trainingStatusInput = document.getElementById('trainingStatus') as HTMLTextAreaElement
const testingStatusInput = document.getElementById('trainingStatus') as HTMLTextAreaElement
const predictInput = document.getElementById('squareFeet') as HTMLInputElement
const returnStatusDiv = document.getElementById('returnStatus') as HTMLDivElement

document.getElementById('bt_train').addEventListener('click', async (event) => {
    await train()
})

document.getElementById('bt_visor').addEventListener('click', async (event) => {
    tfvis.visor().toggle()
})

document.getElementById('bt_test').addEventListener('click', async (event) => {
    await test()
})

document.getElementById('bt_save').addEventListener('click', async (event) => {
    await save()
})

document.getElementById('bt_load').addEventListener('click', async (event) => {
    await load()
})

document.getElementById('bt_predict').addEventListener('click', async (event) => {
    await predict()
})

async function train() {
    const buttonsID = ['bt_train', 'bt_test', 'bt_load', 'bt_save', 'bt_predict']
    buttonsID.forEach(id => {
        document.getElementById(id).setAttribute('disabled', 'disabled')
    })

    trainingStatusInput.value = "Training ....."

    model = createModel()

    const result = await trainModel(model, trainingFeatures, trainingLabel)

    document.getElementById('bt_test').removeAttribute('disabled');
    document.getElementById('bt_save').removeAttribute('disabled');
    document.getElementById('bt_predict').removeAttribute('disabled');
    await plotPredictionLine()

    const loss = result.history.loss.pop()
    const val_loss = result.history.val_loss.pop()
    testingStatusInput.value = "Trainded (unsaved) \n" +
        `Loss: ${loss.toPrecision(5)}\n` +
        `Validation loss: ${val_loss.toPrecision(5)}\n`
}

async function test() {
    console.log(model)
    const loss = Number(await testModel(model, testingFeatures, testingLabel))
    const testingStatusInput = document.getElementById('testingStatus') as HTMLInputElement
    testingStatusInput.value = `Set loss: ${loss.toPrecision(5)}`
}

const storageID = 'house-price-regression'

async function save() {
    const saveResults = await model.save(`localstorage://${storageID}`)
    testingStatusInput.value = `Trainded (saved ${saveResults.modelArtifactsInfo.dateSaved})`
}

async function load() {
    const storageKey = `localstorage://${storageID}`
    const models = await tf.io.listModels()
    const modelInfo = models[storageKey]
    if (modelInfo) {
        model = await tf.loadLayersModel(`localstorage://${storageID}`)

        await plotPredictionLine()
        testingStatusInput.value = `Trainded (loaded ${modelInfo.dateSaved})`
        document.getElementById('bt_test').removeAttribute('disabled');
    } else {
        alert('No saved model found')
    }

}
async function predict(){
    const fPredict = parseInt(predictInput.value)
    if(isNaN(fPredict)){
        alert('Please enter valid number')
    }else {
        tf.tidy(()=>{
            const inputTensor = tf.tensor1d([fPredict])
            const normalizedInput = normalizeTensor(inputTensor,featureTensor.max(),featureTensor.min())
            const normalizedOutputTensor = model.predict(normalizedInput)
            const outputTensor = denormalizeTensor(normalizedOutputTensor,labelTensor.max(),labelTensor.min())

            const value = parseInt((outputTensor.dataSync()[0]/1000).toFixed(0))*1000

            returnStatusDiv.innerHTML = `<h6>The predicted house price <span class="badge bg-secondary">$ ${value}</span></h6>`


        })
    }
}

async function plotPredictionLine(){
    const [xs, ys] = tf.tidy(()=>{
        const normalisedXs = tf.linspace(0,1,100)

        const normalisedYs = model.predict(normalisedXs.reshape([100,1]))

        const xs = denormalizeTensor(normalisedXs,featureTensor.max(),featureTensor.min())
        const ys = denormalizeTensor(normalisedYs,labelTensor.max(),labelTensor.min())

        return [xs.dataSync(),ys.dataSync()]
    })

    const predictPoints = Array.from(xs).map((val,index)=>{
        return {x: val, y: ys[index]}
    })

    await scatterPlot(points,"square",predictPoints)
}
/**
 * Train Model
 * @param model
 * @param trainingFeatureTensor
 * @param trainingLabelTensor
 */
async function trainModel(model, trainingFeatureTensor, trainingLabelTensor) {

    const {onEpochEnd, onBatchEnd} = showFitCallbacks({name: "Training Performance"}, ['loss', 'acc'])

    return model.fit(trainingFeatureTensor, trainingLabelTensor, {
        batchSize: 32,
        epochs: 20,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: onEpochEnd,
            onEpochBegin: async function(){
                await plotPredictionLine()
            }
        }
    });
}

/**
 * Test Model
 * @param model
 * @param testingFeatureTensor
 * @param testingLabelTensor
 */
async function testModel(model, testingFeatureTensor, testingLabelTensor) {
    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
    return await lossTensor.dataSync();
}

/**
 * Create model
 */
function createModel() {
    const model = tf.sequential()
    model.add(tf.layers.dense({
        inputDim: 1,
        activation: "linear",
        units: 1,
        useBias:false
    }))

    const optimizer = tf.train.sgd(0.1)
    model.compile({
        optimizer,
        loss: 'meanSquaredError'
    })

    return model
}


async function start() {

    const data = await tf.data.csv('http://localhost:3000/dist/kc_house_data.csv').toArray()
    console.log('CSV loaded')


    points = data.map((e: { price: string, sqft_living: string }) => {
        return {y: e.price, x: e.sqft_living}
    })

    tf.util.shuffle(points)

    scatterPlot(points,"square")


    if (points.length % 2 !== 0) {
        points.pop()
    }

    const featureValue = points.map(e => e.x)
    const labelValue = points.map(e => e.y)


    featureTensor = tf.tensor(featureValue, [featureValue.length, 1])
    labelTensor = tf.tensor(labelValue, [labelValue.length, 1])


    featureTensorNormalized = normalizeTensor(featureTensor,featureTensor.max(), featureTensor.min())
    labelTensorNormalized = normalizeTensor(labelTensor,labelTensor.max(),labelTensor.min())


    const [testingF, trainingF] = tf.tidy(() => tf.split(featureTensorNormalized, 2))
    const [testingL, trainingL] = tf.tidy(() => tf.split(labelTensorNormalized, 2))

    trainingFeatures = trainingF
    testingFeatures = testingF
    trainingLabel = trainingL
    testingLabel = testingL

    document.getElementById('bt_train').removeAttribute('disabled');

}

console.log('Load CSV ...')
start().then()









