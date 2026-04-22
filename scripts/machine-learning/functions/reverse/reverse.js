import { NumTensor } from "../../tensor.js";

class ReverseFunction {

    /**
     * The write-only activation function
     * 
     * @type {ActivationFunction}
     */
    #composite;

    set composite(composite) {
        this.#composite = composite;
    }

    /**
     * The read-only forwards cache
     * 
     * @type {NumTensor}
     */
    #forwardsCache;

    get forwardsCache() {
        return this.#forwardsCache;
    }

    /**
     * Initialises method hooks to wrap default
     * logic that executes when they are called
     */
    constructor() {
        const step = this.step;

        // The step method hook
        this.step = (optimiser) => {
            step?.call(this, optimiser);
            this.#composite.step(optimiser);
        }

        const forwards = this.forwards;

        // The forwards method hook
        this.forwards = (tensor) => {
            this.#forwardsCache = tensor;
            return forwards.call(this, tensor);
        }
    }

    /**
     * Computes the function application
     * 
     * @param {NumTensor} tensor the input tensor
     * @returns {NumTensor} the function application
     */
    forwards(tensor) {
        return this.#composite.forwards(tensor);
    }

    /**
     * Computes the recursive gradient
     * 
     * @returns {NumTensor} the recursive gradient
     */
    backwards() {
        return this.#composite.backwards();
    }
}

class WeightedReverseFunction extends ReverseFunction {

    get weights() {
        throw new Error("weights getter must be implemented by subclass");
    }

    set weights(weights) {
        throw new Error("weights setter must be implemented by subclass");
    }

    /**
     * The read-only backwards cache
     * 
     * @type {NumTensor}
     */
    #backwardsCache;

    get backwardsCache() {
        return this.#backwardsCache;
    }

    /**
     * Updates the weights using the optimiser
     * 
     * @param {any} optimiser the optimiser
     */
    step(optimiser) {
        this.weights = optimiser.optimise(this.weights, this.backwardsCache);
    }

    backwards() {
        const gradient = super.backwards();

        // Compute the weight gradients of the function
        this.#backwardsCache = this.cacheBackwards(gradient);

        // Return the recursive gradient for the caller
        return gradient;
    }

    /**
     * Computes weight gradients and caches them
     * 
     * @param {NumTensor} gradient the recursive gradient
     */
    cacheBackwards(gradient) {
        throw new Error("cacheBackwards() must be implemented by subclass");
    }
}

export { ReverseFunction, WeightedReverseFunction };