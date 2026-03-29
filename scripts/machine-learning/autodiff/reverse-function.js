import { NumTensor } from "../tensor";

class ReverseFunction {
    /**
     * The cache
     * 
     * @type {NumTensor}
     */
    forwardsCache;
}

class LayerFunction extends ReverseFunction {

    /**
     * The activation function
     * 
     * @type {ActivationFunction}
     */
    #composite;

    /**
     * @param {ActivationFunction} activation
     */
    constructor(composite) {
        this.#composite = composite;
    }

    /**
     * @param {NumTensor} tensor 
     */
    forwards(tensor) {
        this.forwardsCache = tensor;

        return this.#composite.forwards(tensor);
    }

    backwards() {
        return this.#composite.backwards();
    }
}

class LossFunction extends ReverseFunction {

    /**
     * 
     * @param {NumTensor} expected 
     */
    constructor(expected) {
        this.expected = expected;
    }
}

export { LayerFunction, LossFunction };