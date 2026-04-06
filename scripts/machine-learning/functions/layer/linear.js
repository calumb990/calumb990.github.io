import { LayerFunction, ReverseFunction } from "../../autodiff/reverse-function.js";
import { NumTensor } from "../../tensor.js";

export class LinearLayer extends LayerFunction {
    /** @type {NumTensor} */ #weights;

    /** 
     * @param {ReverseFunction} composite 
    */
    constructor(inFeatures, outFeatures) {
        super();

        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;

        this.#weights = new NumTensor([outFeatures, inFeatures]);
        this.#weights._data = this.#weights._data.fill(1);
    }

    forwards(vector) {
        this.forwardsCache = vector;

        return super.forwards(this.#weights.t_mul(vector));
    }
    
    backwards() {
        return this.#weights.permute(1, 0).t_mul(super.backwards());
    }

    cacheBackwards(gradient) {
        return gradient.outer(this.forwardsCache);
    }
}
