import { LossFunction } from "../../autodiff/reverse-function";

// actual vs expected
class MSE extends LossFunction {

    constructor(expected) {
        super(expected);
    }

    forwards(tensor) {
        this.predicted = tensor;

        return Math.pow(tensor - this.expected, 2);
    }

    backwards() {
        return 2 * (this.expected - this.predicted);
    }
}