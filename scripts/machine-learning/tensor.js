export class Tensor {
    /** @type {(tensor: Tensor) => Tensor} */ autograd_fn;

    /**
     * @param {Array<T>} tensor 
     * @param {Boolean} requires_grad
     */
    constructor(tensor, requires_grad = true) {
        this.tensor = tensor;
        this.#validateShape(tensor);
        this.requires_grad = requires_grad;
    }

    /**
     * @param {Array<T>} tensor 
     */
    #validateShape(tensor) {
        this.shape = [tensor.length];

        if (tensor.length < 1) {
            throw new Error("invalid tensor");
        }

        while (Array.isArray(tensor[0])) {
            let end = this.shape.push(tensor[0].length) - 1;

            for (let i = 0; i < tensor.length; i++) {

                if (!Array.isArray(tensor[i])) {
                    throw new Error("invalid tensor");

                } else if (tensor[i].length !== this.shape[end]) {
                    throw new Error("invalid tensor");
                }
            }

            if ((tensor = tensor[0]).length < 1) {
                throw new Error("invalid tensor");
            }
        }
    }
}
