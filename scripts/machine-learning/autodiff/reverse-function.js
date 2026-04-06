import { NumTensor } from "../tensor.js";

class ReverseFunction {}

class LayerFunction extends ReverseFunction {

    /**
     * The activation function
     * 
     * @type {ActivationFunction}
     */
    #composite;

    /**
     * The forwards cache
     * 
     * @type {NumTensor}
     */
    forwardsCache;

    /**
     * The backwards cache
     * 
     * @type {NumTensor}
     */
    backwardsCache;

    set composite(composite) {
        this.#composite = composite;
    }

    /**
     * @param {NumTensor} tensor 
     */
    forwards(tensor) {        
        return this.#composite.forwards(tensor);
    }

    backwards() {
        const gradient = this.#composite.backwards();
        this.backwardsCache = this.cacheBackwards(gradient);
        return gradient;
    }

    cacheBackwards(gradient) {
        return undefined;
    }
}

class LossFunction extends ReverseFunction {

    /**
     * 
     * @param {NumTensor} expected 
     */
    constructor(expected) {
        super();
        this.expected = expected;
    }
}

export { ReverseFunction, LayerFunction, LossFunction };