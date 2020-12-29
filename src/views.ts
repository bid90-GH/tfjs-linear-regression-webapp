import * as tfvis from "@tensorflow/tfjs-vis";

/**
 * Plot points
 * @param points
 * @param featureName
 */
export function scatterPlot(points,featureName, predictPointsArray = null){
    const values = [points]
    const series = ['original']
    if(Array.isArray(predictPointsArray)){
        values.push(predictPointsArray)
        series.push('predicted')
    }

    tfvis.render.scatterplot(
        {name: `${featureName} vs House Price`},
        {values ,series},
        {
            xLabel: featureName,
            yLabel: 'Price',
        }).catch(e=>console.log(e.message))
}

/**
 * Show a summary of the model
 * @param model
 */
export function showModelSummary(model){
    tfvis.show.modelSummary(
        { name: `Model Summary`, tab: `Model` },
        model
    ).catch(e=>console.log(e.message))
}

/**
 * Show a summary of the layer
 */
export function showLayerModel(layer){
    tfvis.show.layer(
        { name: `Layer `, tab: `Model Inspection` },
        layer
    ).catch(e=>console.log(e.message))
}

/**
 * Show fit Callbacks
 */
export function showFitCallbacks(container,metric,opt?){
    return tfvis.show.fitCallbacks(container,metric,opt)
}