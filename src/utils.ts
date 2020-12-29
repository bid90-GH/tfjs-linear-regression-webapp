import * as tf from "@tensorflow/tfjs";


/**
 * Normalize Tensor
 * X'= (X-Xmin)/(Xmax -Xmin)
 * @param tensor
 */

export function normalizeTensor(tensor, t_max, t_min) {
    return tensor.sub(t_min).div(t_max.sub(t_min))
}

/**
 * Normalize Tensor
 *  X = X*(Xmax - Xmin)+Xmin
 * @param tensor
 */
export function denormalizeTensor(normalizedTensor, max, min) {
    return normalizedTensor.mul(max.sub(min)).add(min)
}


