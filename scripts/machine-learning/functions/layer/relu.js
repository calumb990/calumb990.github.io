import { LayerFunction } from "../../autodiff/reverse-function.js";
import { NumTensor } from "../../tensor.js";

export class ReLU extends LayerFunction {

    forwards(tensor) {
        this.forwardsCache = tensor;

        const result = new NumTensor(tensor.shape);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = Math.max(0, tensor._data[i]);
        }

        return super.forwards(result);
    }

    backwards() {
        const result = super.backwards();

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = Math.max(0, result._data[i]);
        }

        debugger;
        return result;
    }
}