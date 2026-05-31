import { NumTensor } from "../../tensor.js";
import { ReverseLossFunction } from "./loss.js";

/**
 * Mean Squared Error (MSE) loss function.
 * 
 * Definition: `(1/n) * sum_{i=1 to n} (ye_i - yp_i)^2`
 */
export class MSE extends ReverseLossFunction {

    /**
     * 
     * @param {NumTensor} predicted 
     * @param {*} expected 
     * @returns 
     */
    apply(predicted, expected) {
        return predicted.sub(expected).pow(2).average;
    }

    /**
     * 
     * @returns 
     */
    backwards() {
        const n = this.predictedCache.shape[0];

        return this.predictedCache.sub(this.expectedCache).s_mul(2 / n);
    }
}
