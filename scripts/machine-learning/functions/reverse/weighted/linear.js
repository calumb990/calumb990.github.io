import { WeightedReverseFunction } from "../reverse.js";
import { NumTensor } from "../../../tensor.js";

/**
 * Linear layer weighted reverse function.
 * 
 * Definition: `f(x) = Wx + b`
 */
export class Linear extends WeightedReverseFunction {

    /**
     * The perceptron weights
     * 
     * @type {NumTensor}
     */ 
    #weights;

    get weights() {
        return this.#weights;
    }

    set weights(weights) {
        this.#weights = weights;
    }

    constructor(inFeatures, outFeatures) {
        super();

        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;

        this.#weights = new NumTensor([outFeatures, inFeatures]);
        this.#weights._data = this.#weights._data.fill(1);
    }

    forwards(tensor) {
        return super.forwards(this.#weights.t_mul(tensor));
    }
    
    backwards() {
        return this.#weights.permute(1, 0).t_mul(super.backwards());
    }

    cacheBackwards(gradient) {
        return gradient.outer(this.forwardsCache);
    }
}
