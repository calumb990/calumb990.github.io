import { ReverseFunction } from "../reverse.js";
import { NumTensor } from "../../../tensor.js";

/**
 * Rectified Linear Unit (ReLU) activation function.
 * 
 * Definition: `f(x) = max(0, x)`
 */
export class ReLU extends ReverseFunction {

    forwards(tensor) {
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

        return result;
    }
}
