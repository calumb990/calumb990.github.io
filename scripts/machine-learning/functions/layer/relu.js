import { LayerFunction } from "../../autodiff/reverse-function";
import { NumTensor } from "../../tensor";

class ReLU extends LayerFunction {

    constructor(composite) {
        super(composite);
    }

    forwards(tensor) {
        const result = new NumTensor(tensor.shape);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = Math.max(0, tensor._data[i]);
        }

        return super.forwards(result);
    }

    backwards() {
        const result = new NumTensor(this.cache.shape);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = this.cache._data[i] <= 0 ? 0 : 1;
        }

        return result.v_mul(super.backwards());
    }
}