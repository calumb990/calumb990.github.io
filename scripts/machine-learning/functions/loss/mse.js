/**
 * Mean Squared Error (MSE) loss function.
 * 
 * Definition: `(1/n) * sum_{i=1 to n} (ye_i - yp_i)^2`
 */
export class MSE {

    constructor(expected) {
        this.expected = expected;
    }

    step() {}

    forwards(tensor) {
        this.predicted = tensor;

        return tensor.sub(this.expected).pow(2).average;
    }

    backwards() {
        const n = this.predicted.shape[0];

        return this.predicted.sub(this.expected).s_mul(2 / n);
    }
}
