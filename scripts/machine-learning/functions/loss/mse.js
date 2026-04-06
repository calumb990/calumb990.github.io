import { LossFunction } from "../../autodiff/reverse-function.js";

export class MSE extends LossFunction {

    constructor(expected) {
        super(expected);
    }

    forwards(tensor) {
        this.predicted = tensor;

        return tensor.sub(this.expected).pow(2).average;
    }

    backwards() {
        const n = this.predicted.shape[0];

        return this.predicted.sub(this.expected).s_mul(2 / n);
    }
}