export class SGD {

    /**
     * @param {number} lr the learning rate 
     */
    constructor(lr) {
        this.lr = lr;
    }

    optimise(weights, gradients) {
        return weights.sub(gradients.s_mul(this.lr));
    }
}
