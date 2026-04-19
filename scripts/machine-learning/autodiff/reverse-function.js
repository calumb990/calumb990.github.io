import { NumTensor } from "../tensor.js";

class ReverseFunction {

    /**
     * The activation function
     * 
     * @type {ActivationFunction}
     */
    #composite;

    set composite(composite) {
        this.#composite = composite;
    }

    /**
     * The forwards cache
     * 
     * @type {NumTensor}
     */
    #forwardsCache;

    get forwardsCache() {
        return this.#forwardsCache;
    }

    /**
     * Defines the forward method hooks
     */
    constructor() {
        const forwards = this.forwards;

        this.forwards = (tensor) => {
            this.#forwardsCache = tensor;
            return forwards.call(this, tensor);
        }
    }

    /**
     * @param {NumTensor} tensor 
     */
    forwards(tensor) {
        return this.#composite.forwards(tensor);
    }

    backwards() {
        return this.#composite.backwards();
    }
}

class WeightedReverseFunction extends ReverseFunction {

    /**
     * The backwards cache
     * 
     * @type {NumTensor}
     */
    #backwardsCache;

    get backwardsCache() {
        return this.#backwardsCache;
    }

    /**
     * 
     * @returns 
     */
    backwards() {
        const gradient = super.backwards();

        // Compute the dL / dw gradients of the function
        this.#backwardsCache = this.cacheBackwards(gradient);

        // Return the dL / da gradients for the caller
        return gradient;
    }

    cacheBackwards(gradient) {
        throw new Error("cacheBackwards() must be implemented by subclass");
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

export { ReverseFunction, WeightedReverseFunction, LossFunction };